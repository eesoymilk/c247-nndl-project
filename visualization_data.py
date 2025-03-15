import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def find_txt_files(root_folder):
    txt_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

def extract_val_cer_range(file_path):
    max_val_cer = None
    min_val_cer = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"'val/CER' reached ([\d\.]+)", line)
            if match:
                val_cer = float(match.group(1))
                if max_val_cer is None or val_cer > max_val_cer:
                    max_val_cer = val_cer
                if min_val_cer is None or val_cer < min_val_cer:
                    min_val_cer = val_cer
    return max_val_cer, min_val_cer

def process_logs_and_plot(log_folder):
    txt_files = find_txt_files(log_folder)
    if not txt_files:
        print("No .txt files found")
        return

    data = {}
    for file_path in txt_files:
        max_val_cer, min_val_cer = extract_val_cer_range(file_path)
        if max_val_cer is not None and min_val_cer is not None:
            file_name = os.path.basename(file_path).replace(".txt", "")
            data[file_name] = (max_val_cer, min_val_cer)

    if not data:
        print("No 'val/CER' data found")
        return
    
    # Adding the baseline data (epoch=0 and epoch=50)
    baseline_max = 1349.82275
    baseline_min = 20.97918
    data['Baseline'] = (baseline_max, baseline_min)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 4))

    models = list(data.keys())
    max_values = [data[m][0] for m in models]
    min_values = [data[m][1] for m in models]
    x = range(len(models))

    width = 0.3
    bars1 = plt.bar([i - width/2 for i in x], max_values, width=width, color="#1f77b4", label="Max val/CER")
    bars2 = plt.bar([i + width/2 for i in x], min_values, width=width, color="#ff7f0e", label="Min val/CER")

    for bar in bars1 + bars2:
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height(), 
                 f"{bar.get_height():.2f}", 
                 ha="center", va="bottom", fontsize=10, color="black")

    plt.xlabel("Model", fontsize=14)
    plt.ylabel("val/CER", fontsize=14)
    plt.title("val/CER per Model (0 & 50 epochs)", fontsize=16, color="darkblue")

    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, max(max_values) * 1.1)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

log_folder = r"D:\ZoStudio\UCLA Yr1\247 Neural Networks and Deep Learning\project\c247-nndl-project\result_logs\data"
process_logs_and_plot(log_folder)
