import re
from pprint import pprint
from pathlib import Path
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_LOGS_DIR = SCRIPT_DIR / "result_logs"
FIGURES_DIR = SCRIPT_DIR / "tex" / "figures"

FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def get_log_file_from_log_dir(log_dir: Path):
    for log_file in log_dir.iterdir():
        if log_file.suffix == ".out":
            return log_file
    raise FileNotFoundError(f"No log file found in {log_dir}")


def get_all_logs(
    result_log_dir: Path = RESULT_LOGS_DIR,
) -> list[tuple[str, Path, Path]]:
    log_dirs = []
    for d in result_log_dir.iterdir():
        if not d.is_dir():
            continue

        if not d.stem.endswith("training"):
            continue

        log_name = d.stem.split("-")[0]

        log_dir = d / "submitit_logs"
        for log_d in log_dir.iterdir():
            if not log_d.is_dir():
                continue
            log_dir = log_d
            break
        else:
            raise FileNotFoundError(f"No log directory found in {d}")
        log_file = get_log_file_from_log_dir(log_dir)
        log_dirs.append((log_name, log_dir, log_file))
    return log_dirs


def get_losses(log_file: Path) -> None:
    pattern = re.compile(r"^Epoch (\d+):.*loss=([0-9\.]+)")
    with open(log_file, "r") as f:
        content = f.readlines()
    epochs = defaultdict(list)
    for line in content:
        match = pattern.search(line)
        if not match:
            continue

        epoch = int(match.group(1))
        loss = float(match.group(2))
        epochs[epoch].append(loss)

    losses = np.array([np.mean(epochs[epoch]) for epoch in sorted(epochs.keys())])
    return losses


def plot_losses(
    log_name: str,
    losses: npt.ArrayLike,
    log_scale: bool = True,
    save_path: Path = FIGURES_DIR,
) -> None:
    filename = f"{log_name}_loss"

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.xlim(0, len(losses))

    if log_scale:
        plt.yscale("log")
        plt.ylabel("Loss (Log Scale)")
        filename += "_log"
    else:
        plt.ylabel("Loss")

    plt.title(f"Loss for {log_name}")
    plt.tight_layout()
    plt.savefig(save_path / f"{filename}.png")
    plt.close()
    print(f"Saved plot: {save_path / filename}")


def main() -> None:
    log_dirs = get_all_logs()
    for log_name, log_dir, log_file in log_dirs:
        print(f"Log Name: {log_name}")
        print(f"Log Dir: {log_dir}")
        print(f"Log File: {log_file}")
        losses = get_losses(log_file)
        plot_losses(log_name, losses, log_scale=False)
        plot_losses(log_name, losses, log_scale=True)


if __name__ == "__main__":
    main()
