#!/usr/bin/env python3
"""
RAG系統自動化打包腳本
這個腳本會自動執行所有必要的步驟來打包RAG系統為可執行文件
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
import time

def print_step(step_name, step_number, total_steps):
    """打印步驟信息"""
    print(f"\n{'='*60}")
    print(f"步驟 {step_number}/{total_steps}: {step_name}")
    print(f"{'='*60}")

def run_command(command, check=True, shell=True):
    """執行命令並返回結果"""
    print(f"執行命令: {command}")
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            print(f"輸出: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"命令執行失敗: {e}")
        if e.stderr:
            print(f"錯誤: {e.stderr}")
        return None

def check_python_version():
    """檢查Python版本"""
    print_step("檢查Python版本", 1, 8)
    
    version = sys.version_info
    print(f"當前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    print("✅ Python版本檢查通過")
    return True

def check_system_requirements():
    """檢查系統需求"""
    print_step("檢查系統需求", 2, 8)
    
    # 檢查磁碟空間
    disk_usage = shutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    print(f"可用磁碟空間: {free_gb:.2f} GB")
    
    if free_gb < 5:
        print("❌ 需要至少5GB可用磁碟空間")
        return False
    
    # 檢查作業系統
    os_name = platform.system()
    print(f"作業系統: {os_name}")
    
    print("✅ 系統需求檢查通過")
    return True

def install_dependencies():
    """安裝依賴"""
    print_step("安裝依賴", 3, 8)
    
    # 升級pip
    print("升級pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # 安裝requirements
    if os.path.exists("requirements.txt"):
        print("安裝requirements.txt中的依賴...")
        result = run_command(f"{sys.executable} -m pip install -r requirements.txt")
        if result is None:
            print("❌ 依賴安裝失敗")
            return False
    else:
        print("❌ 找不到requirements.txt文件")
        return False
    
    print("✅ 依賴安裝完成")
    return True

def skip_dependencies():
    """跳過依賴安裝"""
    print_step("跳過依賴安裝", 3, 8)
    print("⚠️  跳過依賴安裝步驟")
    print("假設當前環境已經安裝了所有必要的依賴")
    print("✅ 依賴檢查跳過")
    return True

def check_required_files():
    """檢查必要文件"""
    print_step("檢查必要文件", 4, 8)
    
    required_files = [
        "qa_RAGsystem_printChunk_QAsplit_coquiTTSclone.py",
        "build_config.spec",
        "requirements.txt"
    ]
    
    required_dirs = [
        "technical_file",
        "voice"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for dir in required_dirs:
        if not os.path.exists(dir):
            missing_files.append(dir)
    
    if missing_files:
        print(f"❌ 缺少必要文件/目錄: {missing_files}")
        return False
    
    print("✅ 必要文件檢查通過")
    return True

def clean_build_files():
    """清理舊的建置文件"""
    print_step("清理舊的建置文件", 5, 8)
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    files_to_clean = ["*.pyc", "*.pyo"]
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"刪除目錄: {dir_name}")
            shutil.rmtree(dir_name)
    
    # 清理緩存文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"刪除文件: {file_path}")
                except OSError:
                    pass
    
    print("✅ 舊文件清理完成")
    return True

def test_main_script():
    """測試主程序是否能正常導入"""
    print_step("測試主程序", 6, 8)
    
    try:
        # 嘗試導入主模組檢查語法
        print("檢查主程序語法...")
        result = run_command(f"{sys.executable} -m py_compile qa_RAGsystem_printChunk_QAsplit_coquiTTSclone.py")
        
        if result is None:
            print("❌ 主程序語法檢查失敗")
            return False
        
        print("✅ 主程序語法檢查通過")
        return True
    
    except Exception as e:
        print(f"❌ 主程序測試失敗: {e}")
        return False

def build_executable(use_simple_config=False):
    """執行打包"""
    print_step("開始打包", 7, 8)
    
    print("開始使用PyInstaller打包...")
    print("⚠️  這個過程可能需要10-30分鐘，請耐心等待...")
    
    start_time = time.time()
    
    # 步驟 4: 使用 PyInstaller 打包
    print(f"\n{'-'*50}")
    print("步驟 4: 使用 PyInstaller 打包")
    print(f"{'-'*50}")
    
    # 使用完整配置文件
    spec_file = "build_config.spec"
    print(f"使用完整配置文件: {spec_file}")
    
    # 檢查配置文件是否存在
    if not os.path.exists(spec_file):
        print(f"❌ 錯誤：配置文件 {spec_file} 不存在！")
        return False
    
    try:
        print(f"正在使用 {spec_file} 進行打包...")
        subprocess.run([
            "pyinstaller", 
            "--clean",  # 清理之前的建置
            spec_file
        ], check=True)
        
        print("✅ PyInstaller 打包完成！")
        
        # 檢查生成的執行文件
        exe_path = "dist/RAG_System.exe"
            
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path) / (1024 * 1024 * 1024)  # GB
            print(f"✅ 執行文件已生成: {exe_path}")
            print(f"📁 文件大小: {file_size:.2f} GB")
        else:
            print(f"⚠️ 執行文件未找到: {exe_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ PyInstaller 打包失敗: {e}")
        return False
    
    end_time = time.time()
    build_time = end_time - start_time
    print(f"✅ 打包完成，耗時: {build_time:.2f}秒")
    return True

def finalize_build():
    """完成打包並提供說明"""
    print_step("完成打包", 8, 8)
    
    exe_path = None
    
    # 尋找生成的可執行文件
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            if file.endswith(".exe") or (not file.endswith(".exe") and os.path.isfile(os.path.join("dist", file))):
                exe_path = os.path.join("dist", file)
                break
    
    if exe_path and os.path.exists(exe_path):
        file_size = os.path.getsize(exe_path) / (1024**2)  # MB
        print(f"✅ 可執行文件已生成: {exe_path}")
        print(f"✅ 文件大小: {file_size:.2f} MB")
        
        print("\n" + "="*60)
        print("打包完成！使用說明:")
        print("="*60)
        print(f"1. 可執行文件位置: {exe_path}")
        print("2. 首次運行前請確保:")
        print("   - Ollama服務正在運行")
        print("   - 相應的LLM模型已下載")
        print("   - 音頻設備工作正常（如果使用語音功能）")
        print("3. 運行時可能需要:")
        print("   - 網路連接（下載TTS/Whisper模型）")
        print("   - 足夠的RAM（建議8GB以上）")
        print("4. 如果遇到問題，可以:")
        print("   - 檢查日誌輸出")
        print("   - 確認所有依賴服務正常運行")
        print("   - 查看README文件中的故障排除指南")
        print("="*60)
        
        return True
    else:
        print("❌ 未找到生成的可執行文件")
        return False

def main():
    """主函數"""
    print("RAG系統自動化打包腳本")
    print("=" * 60)
    
    print("選擇打包配置：")
    print("1. 完整配置 (build_config.spec)")
    print("2. 跳過依賴安裝，使用完整配置")
    
    choice = input("請選擇 (1-2): ").strip()
    
    skip_deps = choice == '2'
    use_simple_config = False  # 移除簡化配置選項
    
    if not skip_deps:
        print("\n✅ 已選擇完整打包模式（包含依賴安裝）")
    else:
        print("\n✅ 已選擇快速打包模式（跳過依賴安裝）")
    
    print("✅ 使用完整配置")
    
    start_time = time.time()
    
    # 執行所有步驟
    steps = [
        check_python_version,
        check_system_requirements,
        skip_dependencies if skip_deps else install_dependencies,
        check_required_files,
        clean_build_files,
        test_main_script,
        lambda: build_executable(use_simple_config),  # 傳遞參數
        finalize_build
    ]
    
    for i, step in enumerate(steps, 1):
        if not step():
            print(f"\n❌ 打包失敗，在步驟 {i} 中止")
            sys.exit(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 打包完全成功！總耗時: {total_time:.2f}秒")
    print("您現在可以分發和運行生成的可執行文件了！")

if __name__ == "__main__":
    main() 