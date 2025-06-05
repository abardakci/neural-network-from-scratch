import subprocess
import os
import sys

# Bu betiğin (çalıştırdığınız betiğin) dizinini alın
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# main.py dosyasının mutlak yolunu oluşturun
main_py_path = os.path.join(current_script_dir, "src", "main.py")

# Şimdi bu yolu kullanarak çalıştırın
subprocess.run([sys.executable, main_py_path])