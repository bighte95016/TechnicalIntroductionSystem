#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Ollama 模組診斷和修復腳本
用於解決 PyInstaller 打包時 langchain_ollama 模組缺失的問題
"""

import sys
import subprocess
import importlib
import pkgutil
import os
from datetime import datetime

def check_module_installation(module_name):
    """檢查模組是否已安裝"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} 已安裝")
        return True
    except ImportError as e:
        print(f"❌ {module_name} 未安裝: {e}")
        return False

def get_module_submodules(module_name):
    """獲取模組的所有子模組"""
    try:
        module = importlib.import_module(module_name)
        submodules = []
        
        if hasattr(module, '__path__'):
            for importer, modname, ispkg in pkgutil.iter_modules(module.__path__, module.__name__ + "."):
                submodules.append(modname)
                print(f"  📦 發現子模組: {modname}")
                
                # 遞歸查找子模組的子模組
                if ispkg:
                    try:
                        sub_submodules = get_module_submodules(modname)
                        submodules.extend(sub_submodules)
                    except:
                        pass
        
        return submodules
    except Exception as e:
        print(f"❌ 獲取 {module_name} 子模組時出錯: {e}")
        return []

def install_missing_packages():
    """安裝缺失的套件"""
    packages_to_install = [
        'langchain-ollama',
        'langchain-core',
        'langchain-community',
        'ollama'
    ]
    
    print("正在安裝缺失的套件...")
    for package in packages_to_install:
        try:
            print(f"安裝 {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package, '--upgrade'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ {package} 安裝成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安裝失敗: {e}")
            print(f"錯誤輸出: {e.stderr}")

def generate_hiddenimports_list():
    """生成完整的 hiddenimports 列表"""
    modules_to_check = [
        'langchain_ollama',
        'langchain_core', 
        'langchain_community',
        'ollama'
    ]
    
    all_hiddenimports = set()
    
    for module in modules_to_check:
        print(f"\n🔍 分析模組: {module}")
        if check_module_installation(module):
            submodules = get_module_submodules(module)
            all_hiddenimports.add(module)
            all_hiddenimports.update(submodules)
    
    print(f"\n📝 生成的 hiddenimports 列表 (共 {len(all_hiddenimports)} 個模組):")
    print("hiddenimports = [")
    for module in sorted(all_hiddenimports):
        print(f"    '{module}',")
    print("]")
    
    # 保存到文件
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('langchain_hiddenimports.txt', 'w', encoding='utf-8') as f:
        f.write("# LangChain Ollama 相關 hiddenimports\n")
        f.write(f"# 生成時間: {current_time}\n\n")
        for module in sorted(all_hiddenimports):
            f.write(f"'{module}',\n")
    
    print(f"\n💾 hiddenimports 列表已保存到 langchain_hiddenimports.txt")

def test_imports():
    """測試關鍵模組的導入"""
    test_modules = [
        'langchain_ollama',
        'langchain_ollama.llms',
        'langchain_ollama.embeddings',
        'langchain_ollama.chat_models',
        'langchain_core',
        'ollama'
    ]
    
    print("\n🧪 測試關鍵模組導入:")
    success_count = 0
    
    for module in test_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} 導入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module} 導入失敗: {e}")
    
    print(f"\n📊 導入測試結果: {success_count}/{len(test_modules)} 成功")
    return success_count == len(test_modules)

def check_environment():
    """檢查 Python 環境"""
    print("🔍 檢查 Python 環境:")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路徑: {sys.executable}")
    print(f"當前工作目錄: {os.getcwd()}")
    
    # 檢查 pip 版本
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"pip 版本: {result.stdout.strip()}")
    except:
        print("❌ 無法獲取 pip 版本")

def main():
    """主函數"""
    print("🚀 LangChain Ollama 模組診斷和修復工具")
    print("=" * 50)
    
    # 檢查環境
    check_environment()
    
    print("\n" + "=" * 50)
    
    # 檢查關鍵模組
    print("📋 檢查關鍵模組安裝狀態:")
    critical_modules = [
        'langchain_ollama',
        'langchain_core',
        'langchain_community',
        'ollama'
    ]
    
    missing_modules = []
    for module in critical_modules:
        if not check_module_installation(module):
            missing_modules.append(module)
    
    # 如果有缺失模組，嘗試安裝
    if missing_modules:
        print(f"\n⚠️ 發現 {len(missing_modules)} 個缺失模組")
        install_choice = input("是否自動安裝缺失的套件? (y/n): ").strip().lower()
        if install_choice == 'y':
            install_missing_packages()
        else:
            print("跳過自動安裝，請手動安裝以下套件:")
            for module in missing_modules:
                print(f"  pip install {module.replace('_', '-')}")
    
    print("\n" + "=" * 50)
    
    # 測試導入
    if test_imports():
        print("\n🎉 所有關鍵模組導入成功！")
        
        # 生成 hiddenimports 列表
        generate_choice = input("\n是否生成完整的 hiddenimports 列表? (y/n): ").strip().lower()
        if generate_choice == 'y':
            generate_hiddenimports_list()
        
    else:
        print("\n❌ 部分模組導入失敗，請檢查安裝狀態")
    
    print("\n" + "=" * 50)
    print("🏁 診斷完成")

if __name__ == "__main__":
    main() 