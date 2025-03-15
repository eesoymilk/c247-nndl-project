import os
import re

def generate_txt_files(log_file_folder):
    for root, _, files in os.walk(log_file_folder):
        for file in files:
            if file.endswith("log.out"):
                err_path = os.path.join(root, file)
                txt_path = os.path.join(root, file.replace(".out", ".txt"))

                try:
                    with open(err_path, "r", encoding="utf-8") as f:
                        content = f.readlines()

                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.writelines(content)

                    print(f"Converted: {err_path} -> {txt_path}")

                except Exception as e:
                    print(f"Failed to process {err_path}: {e}")

log_file_folder = "result_logs/testing"
generate_txt_files(log_file_folder)
