# Import the subprocess module to run external commands
import subprocess
import sys

def test_espeak(text, language_code=None, voice_variant=None, speed=None, pitch=None):
    """
    Uses the espeak-ng command-line tool to synthesize speech.

    Args:
        text (str): The text to speak.
        language_code (str, optional): Language code (e.g., 'en', 'zh'). Defaults to None.
        voice_variant (str, optional): Specific voice variant (e.g., 'f3', 'm1'). Defaults to None.
        speed (int, optional): Speed in words per minute (e.g., 140). Defaults to None (espeak default).
        pitch (int, optional): Pitch adjustment (0-99, 50 is default). Defaults to None.
    """
    try:
        # Use 'espeak-ng' as it's the common name for the modern version
        command = ['espeak-ng']
        
        # Construct the voice argument (-v)
        if language_code:
            voice_arg = language_code
            if voice_variant:
                voice_arg += f"+{voice_variant}"
            command.extend(['-v', voice_arg])
            
        # Add amplitude (volume) - Try reducing to avoid clipping/noise
        command.extend(['-a', '50']) # Reduced from 150, try 100 or 90
        
        # Add speed if specified
        if speed is not None:
            command.extend(['-s', str(speed)]) 
            
        # Add pitch if specified
        if pitch is not None:
            command.extend(['-p', str(pitch)])
            
        # Add the text to speak
        encoding = sys.getdefaultencoding() if sys.platform != 'win32' else 'utf-8'
        
        print(f"\nExecuting command: {' '.join(command)} \"{text[:50]}...\"")
        
        # Run the command
        process = subprocess.run(
            command, 
            input=text, 
            text=True, 
            check=True, 
            capture_output=True, 
            encoding=encoding
        )
        
        print(f"eSpeak ({language_code or 'default'}) finished successfully.")
        # Optional: print output if needed
        # if process.stdout: print("eSpeak stdout:", process.stdout)
        # if process.stderr: print("eSpeak stderr:", process.stderr)

    except FileNotFoundError:
        print("\nError: 'espeak-ng' command not found.")
        print("Please ensure eSpeak-NG is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during eSpeak execution:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        # Print stderr which often contains useful error details from espeak
        if e.stderr:
             print(f"Stderr: {e.stderr.strip()}")
        # Also print stdout if any
        if e.stdout:
            print(f"Stdout: {e.stdout.strip()}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    print("--- Testing eSpeak with adjusted parameters ---")

    # 1. Test English with adjusted speed and default pitch
    english_text = "Hello, this is a test of the eSpeak engine in English, with adjusted speed."
    test_espeak(english_text, language_code='en', speed=145) 

    # 2. Test English with adjusted speed and slightly lower pitch
    english_text_pitch = "Testing English again, this time also adjusting the pitch slightly lower."
    test_espeak(english_text_pitch, language_code='en', speed=145, pitch=45)

    # 3. Test Chinese (Mandarin) with adjusted speed
    # Ensure 'zh' (or appropriate code from --voices=zh) works on your system
    chinese_text = "你好，这是一个调整了语速的中文语音合成测试。"
    test_espeak(chinese_text, language_code='zh', speed=200, pitch=70) # Chinese might need slower speed

    # 4. Example: Test Chinese with a specific (hypothetical) voice variant 
    # chinese_text_variant = "尝试使用一个特定的中文语音变体。"
    # Find available variants using `espeak-ng --voices=zh`
    # Replace 'm1' with an actual variant like f5, m3 etc. if available
    # test_espeak(chinese_text_variant, language_code='zh', voice_variant='m1', speed=140)

    print("\n--- eSpeak Testing Complete ---")