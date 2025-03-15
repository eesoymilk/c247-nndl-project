import os
import re
import numpy as np
import matplotlib.pyplot as plt


def convert_and_plot(log_file_folder):
    for root, _, files in os.walk(log_file_folder):
        for file in files:
            if file.endswith("log.err"):
                err_path = os.path.join(root, file)
                txt_path = os.path.join(root, file.replace(".err", ".txt"))

                epochs = []
                val_cers = []

                try:
                    with open(err_path, "r", encoding="utf-8") as f:
                        content = f.readlines()

                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.writelines(content)

                    print(f"Converted: {err_path} -> {txt_path}")

                    for line in content:
                        epoch_match = re.search(r"Epoch (\d+)", line)
                        val_cer_match = re.search(r"'val/CER' reached ([0-9\.]+)", line)

                        if epoch_match and val_cer_match:
                            epoch = int(epoch_match.group(1))
                            val_cer = float(val_cer_match.group(1))

                            epochs.append(epoch)
                            val_cers.append(val_cer)

                    if epochs and val_cers:
                        # NOTE: log
                        log_val_cers = np.log1p(val_cers)

                        plt.figure()
                        plt.plot(epochs, log_val_cers, marker="o", linestyle="-")
                        plt.xlabel("Epoch")
                        plt.ylabel("Log(Validation CER)")
                        plt.title(f"Validation CER for {file} (Log Scale)")
                        plt.grid()
                        plt.show()

                        # plot_path = os.path.join(root, file.replace(".err", ".png"))
                        # plt.savefig(plot_path)
                        # plt.close()

                        # print(f"Saved plot: {plot_path}")

                except Exception as e:
                    print(f"Failed to process {err_path}: {e}")


log_file_folder = "result_logs"
convert_and_plot(log_file_folder)
