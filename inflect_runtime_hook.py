# PyInstaller runtime hook for inflect and typeguard
# 解決源碼檢查問題

import sys
import os

def patch_inspect_early():
    """提前修補inspect.getsource，避免任何模組載入時的源碼檢查問題"""
    try:
        import inspect
        
        # 保存原始函數
        if not hasattr(inspect, '_original_getsource'):
            inspect._original_getsource = inspect.getsource
            inspect._original_getsourcelines = inspect.getsourcelines
            inspect._original_findsource = inspect.findsource
            
            def safe_getsource(obj):
                """安全的getsource，失敗時返回空字符串"""
                try:
                    return inspect._original_getsource(obj)
                except (OSError, TypeError, IOError):
                    return "# Source code not available in packaged application"
            
            def safe_getsourcelines(obj):
                """安全的getsourcelines，失敗時返回空列表"""
                try:
                    return inspect._original_getsourcelines(obj)
                except (OSError, TypeError, IOError):
                    return (["# Source code not available in packaged application\n"], 1)
            
            def safe_findsource(obj):
                """安全的findsource，失敗時返回空結果"""
                try:
                    return inspect._original_findsource(obj)
                except (OSError, TypeError, IOError):
                    return (["# Source code not available in packaged application\n"], 1)
            
            # 替換函數
            inspect.getsource = safe_getsource
            inspect.getsourcelines = safe_getsourcelines
            inspect.findsource = safe_findsource
            
        print("✅ inspect模組已提前修補")
        
    except Exception as e:
        print(f"⚠️ 修補inspect時發生錯誤: {e}")

def patch_typeguard_later():
    """在模組載入後修補typeguard"""
    try:
        # 檢查typeguard是否已經載入
        if 'typeguard' in sys.modules:
            import typeguard
            
            # 禁用typeguard的類型檢查
            if hasattr(typeguard, 'typechecked') and not hasattr(typeguard, '_patched'):
                original_typechecked = typeguard.typechecked
                
                def dummy_typechecked(func=None, **kwargs):
                    """空的typechecked裝飾器，避免源碼檢查"""
                    if func is None:
                        return lambda f: f
                    return func
                
                typeguard.typechecked = dummy_typechecked
                typeguard._patched = True
                print("✅ typeguard已修補")
                
    except Exception as e:
        print(f"⚠️ 修補typeguard時發生錯誤: {e}")

# 立即執行inspect修補
patch_inspect_early()

# 設置模組載入後的hook
if hasattr(sys, 'meta_path'):
    class TypeguardPatcher:
        def find_spec(self, fullname, path, target=None):
            if fullname == 'typeguard':
                # 當typeguard被載入時，設置一個延遲修補
                import threading
                def delayed_patch():
                    import time
                    time.sleep(0.1)  # 短暫延遲確保模組完全載入
                    patch_typeguard_later()
                
                thread = threading.Thread(target=delayed_patch)
                thread.daemon = True
                thread.start()
            return None
    
    # 添加到meta_path的開頭
    sys.meta_path.insert(0, TypeguardPatcher()) 