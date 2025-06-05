import subprocess
import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))

main_py_path = os.path.join(current_script_dir, "src", "main.py")

subprocess.run([sys.executable, main_py_path])