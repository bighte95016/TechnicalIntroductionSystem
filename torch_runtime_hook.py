# PyInstaller runtime hook for PyTorch
# 解決TorchScript編譯問題

import os
import sys

# 設置環境變量來禁用PyTorch JIT
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_JIT_DISABLE'] = '1'

# 嘗試禁用PyTorch的JIT功能
try:
    import torch
    
    # 禁用JIT編譯
    if hasattr(torch.jit, 'set_fusion_strategy'):
        torch.jit.set_fusion_strategy([('STATIC', 0), ('DYNAMIC', 0)])
    
    # 禁用性能分析 - 安全檢查
    if hasattr(torch, '_C') and hasattr(torch._C, '_jit_set_profiling_mode'):
        torch._C._jit_set_profiling_mode(False)
    
    if hasattr(torch, '_C') and hasattr(torch._C, '_jit_set_profiling_executor'):
        torch._C._jit_set_profiling_executor(False)
    
    # 禁用TorchScript優化 - 安全檢查屬性是否存在
    if hasattr(torch.jit, '_state'):
        state_module = torch.jit._state
        if hasattr(state_module, 'disable_jit_autocast'):
            state_module.disable_jit_autocast()
        else:
            print("警告：torch.jit._state.disable_jit_autocast 不存在，跳過此配置")
    
    print("PyTorch JIT功能已禁用")
    
except ImportError:
    print("PyTorch未安裝，跳過JIT配置")
except Exception as e:
    print(f"配置PyTorch JIT時發生錯誤: {e}")
    # 不要因為這個錯誤而中斷程序
    pass 