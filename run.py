import subprocess
import os
import sys
import zipfile

current_script_dir = os.path.dirname(os.path.abspath(__file__))

main_py_path = os.path.join(current_script_dir, "src", "main.py")

mnist_zip_path = os.path.join(current_script_dir, "mnist.zip")
mnist_train_csv_path = os.path.join(current_script_dir, "mnist_train.csv")
mnist_test_csv_path = os.path.join(current_script_dir, "mnist_test.csv")

if not os.path.exists(mnist_train_csv_path) or not os.path.exists(mnist_test_csv_path):
    print("mnist.zip unzipping...")
    
    try:
        with zipfile.ZipFile(mnist_zip_path, 'r') as zip_ref:
            zip_ref.extract('mnist_train.csv', path=current_script_dir)
            print("mnist_train.csv extracted.")
        
            zip_ref.extract('mnist_test.csv', path=current_script_dir)
            print("mnist_test.csv extracted.")


    except Exception as e:
        print(f"Error while extracting: {e}")
        sys.exit(1)

# main.py
print(f"'{main_py_path}' executing...")
result = subprocess.run([sys.executable, main_py_path], capture_output=True, text=True)
