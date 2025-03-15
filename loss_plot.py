import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_LOGS_DIR = SCRIPT_DIR / "result_logs" / "training"
FIGURES_DIR = SCRIPT_DIR / "tex" / "figures"

FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def get_log_dir(parent_log_dir: Path):
    log_dir = parent_log_dir / "submitit_logs"
    for log_d in log_dir.iterdir():
        if not log_d.is_dir():
            continue
        return log_d
    raise FileNotFoundError(f"No log directory found in {log_dir}")


def get_log_file_from_log_dir(log_dir: Path):
    for log_file in log_dir.iterdir():
        if log_file.suffix == ".out":
            return log_file
    raise FileNotFoundError(f"No log file found in {log_dir}")


def get_all_logs(
    result_log_dir: Path = RESULT_LOGS_DIR,
) -> list[tuple[str, npt.ArrayLike]]:
    tmp_logs: list[tuple[str, npt.ArrayLike]] = []

    for d in result_log_dir.iterdir():
        if not d.is_dir():
            continue

        if not d.stem.endswith("-training"):
            continue

        log_name = d.stem.split("-")[0]

        log_dir = get_log_dir(d)
        log_file = get_log_file_from_log_dir(log_dir)
        loss_history = get_loss_history(log_file)
        tmp_logs.append((log_name, loss_history))

    # append log name with _200 into the one without
    logs: list[tuple[str, npt.ArrayLike]] = []
    for log_name_200, loss_history_200 in tmp_logs:
        if not log_name_200.endswith("_200"):
            continue

        log_name = log_name_200.replace("_200", "")
        for name, loss_history in tmp_logs:
            if name != log_name:
                continue
            loss_history = np.concatenate([loss_history, loss_history_200])
            logs.append((log_name, loss_history))
            break

    return logs


def get_loss_history(log_file: Path) -> None:
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


def plot_all_losses(
    losses: list[tuple[str, npt.ArrayLike]],
    save_path: Path = FIGURES_DIR,
    ignore_first: bool = False,
    filename: str = "all_losses",
) -> None:
    plt.figure()
    for name, loss_history in losses:
        if ignore_first:
            loss_history = loss_history[1:]
        plt.plot(loss_history, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.title("Loss Comparison")
    plt.tight_layout()

    out_file = save_path / f"{filename}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved combined losses plot to: {out_file}")


def main() -> None:
    losses = get_all_logs()
    plot_all_losses(losses, ignore_first=True)


if __name__ == "__main__":
    main()
