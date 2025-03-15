import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def convert_and_plot(log_file_folder):
    plt.figure(figsize=(12, 6))

    model_data = {}

    for root, _, files in os.walk(log_file_folder):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                
                if file.endswith("100.txt") or file.endswith("200.txt"):
                    prefix = file[:-7]  
                    
                    if prefix not in model_data:
                        model_data[prefix] = {"epochs": [], "val_cers": [], "max_epoch": 0}
                    
                    epochs = []
                    val_cers = []

                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.readlines()

                        for line in content:
                            epoch_match = re.search(r"Epoch (\d+)", line)
                            val_cer_match = re.search(r"'val/CER' reached ([0-9\.]+)", line)

                            if epoch_match and val_cer_match:
                                epoch = int(epoch_match.group(1))
                                val_cer = float(val_cer_match.group(1))

                                epochs.append(epoch)
                                val_cers.append(val_cer)

                        if file.endswith("200.txt"):
                            max_epoch = model_data[prefix]["max_epoch"]  
                            epochs = [epoch + max_epoch for epoch in epochs]

                        if epochs and val_cers:
                            model_data[prefix]["epochs"].extend(epochs)
                            model_data[prefix]["val_cers"].extend(val_cers)

                            model_data[prefix]["max_epoch"] = max(epochs)

                    except Exception as e:
                        print(f"Failed to process {txt_path}: {e}")
    
    if not model_data:
        print("No valid data found for plotting.")
        return

    color_cycle = plt.cm.get_cmap('tab10', len(model_data)) 
    style = ['-', '--', '-.', ':'] 


    for idx, (prefix, data) in enumerate(model_data.items()):
        epochs = data["epochs"]
        val_cers = data["val_cers"]
        
        if len(val_cers) > 0:
            unique_epochs, unique_val_cers = [], []
            for epoch, val_cer in zip(epochs, val_cers):
                if epoch not in unique_epochs:
                    unique_epochs.append(epoch)
                    unique_val_cers.append(val_cer)
                else:
                    idx = unique_epochs.index(epoch)
                    unique_val_cers[idx] = np.mean([unique_val_cers[idx], val_cer])

            epochs = unique_epochs
            val_cers = unique_val_cers

            # NOTE: log
            log_val_cers = np.log1p(val_cers)

            sorted_indices = np.argsort(epochs)
            epochs = np.array(epochs)[sorted_indices]
            log_val_cers = np.array(log_val_cers)[sorted_indices]

            try:
                cs = CubicSpline(epochs, log_val_cers)

                smooth_epochs = np.linspace(min(epochs), max(epochs), 500)
                smooth_log_val_cers = cs(smooth_epochs)

                color = color_cycle(idx)
                linestyle = style[idx % len(style)]

                plt.plot(smooth_epochs, smooth_log_val_cers, linestyle=linestyle, label=f"{prefix} (log scale)", 
                         linewidth=2, color=color)

            except ValueError as e:
                print(f"Error with cubic spline interpolation for model {prefix}: {e}")

    plt.xlabel("Epoch", fontsize=14, weight='bold')
    plt.ylabel("Log(Validation CER)", fontsize=14, weight='bold')
    plt.title("Validation CER for All Models (Log Scale)", fontsize=16, weight='bold')

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(fontsize=12, loc='best')

    plt.gcf().set_facecolor('#f5f5f5')

    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)

    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()

log_file_folder = "result_logs/training" 
convert_and_plot(log_file_folder)
