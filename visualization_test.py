import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics_from_file(txt_path):
    val_cer, test_cer, val_loss, test_loss = None, None, None, None
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.readlines()
        
        for line in content:
            val_cer_match = re.search(r"val/CER\s+([0-9\.]+)", line)
            test_cer_match = re.search(r"test/CER\s+([0-9\.]+)", line)
            val_loss_match = re.search(r"val/loss\s+([0-9\.]+)", line)
            test_loss_match = re.search(r"test/loss\s+([0-9\.]+)", line)

            if val_cer_match:
                val_cer = float(val_cer_match.group(1))
            if test_cer_match:
                test_cer = float(test_cer_match.group(1))
            if val_loss_match:
                val_loss = float(val_loss_match.group(1))
            if test_loss_match:
                test_loss = float(test_loss_match.group(1))

        return val_cer, test_cer, val_loss, test_loss

    except Exception as e:
        print(f"Failed to process {txt_path}: {e}")
        return None, None, None, None

def plot_metrics(log_file_folder):
    models = []
    val_cers = []
    test_cers = []
    val_losses = []
    test_losses = []

    for root, _, files in os.walk(log_file_folder):
        for file in files:
            if file.endswith(".txt"):
                model_name = os.path.splitext(file)[0]
                txt_path = os.path.join(root, file)

                val_cer, test_cer, val_loss, test_loss = extract_metrics_from_file(txt_path)

                if val_cer is not None and test_cer is not None and val_loss is not None and test_loss is not None:
                    models.append(model_name)
                    val_cers.append(val_cer)
                    test_cers.append(test_cer)
                    val_losses.append(val_loss)
                    test_losses.append(test_loss)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    fig.patch.set_facecolor('#f7f7f7')

    x_pos = np.arange(len(models))
    ax1.bar(x_pos - 0.15, val_cers, width=0.3, label="val/CER", color="#1f77b4", edgecolor="black", linewidth=1.5)
    ax1.bar(x_pos + 0.15, test_cers, width=0.3, label="test/CER", color="#ff7f0e", edgecolor="black", linewidth=1.5)
    
    ax1.set_xlabel("Model", fontsize=14, weight="bold")
    ax1.set_ylabel("CER", fontsize=14, weight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, fontsize=12, weight="bold")
    ax1.set_title("CER and Loss Metrics for Different Models", fontsize=16, weight="bold")

    ax1.legend(loc='upper left', fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(models, val_losses, color='#2ca02c', marker='o', label="val/loss", linestyle='-', linewidth=2, markersize=8)
    ax2.plot(models, test_losses, color='#d62728', marker='s', label="test/loss", linestyle='-', linewidth=2, markersize=8)

    ax2.set_ylabel("Loss", fontsize=14, weight="bold")

    ax2.legend(loc='upper right', fontsize=12)

    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(False) 

    plt.tight_layout()

    plt.show()

log_file_folder = "result_logs/testing"
plot_metrics(log_file_folder)
