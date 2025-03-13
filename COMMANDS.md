## 

### Baseline TDS model

- Training
    ```bash
    python -m emg2qwerty.train \
        user=single_user \
        model=tds_conv_ctc \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        --multirun
    ```
- Testing
    ```bash
    python -m emg2qwerty.train \
        user=single_user \
        model=tds_conv_ctc \
        train=False \
        trainer.accelerator=gpu \
        decoder=ctc_greedy \
        hydra.launcher.mem_gb=64 \
        --multirun
    ```

### Baseline TDS Model with Data Preprocessing

- Training
    ```bash
    python -m emg2qwerty.train \
        user=single_user \
        model=tds_conv_ctc \
        transforms=custom \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        --multirun
    ```
- Testing
    ```bash
    python -m emg2qwerty.train \
        user=single_user \
        model=tds_conv_ctc \
        transforms=custom \
        train=False \
        trainer.accelerator=gpu \
        decoder=ctc_greedy \
        hydra.launcher.mem_gb=64 \
        --multirun
    ```

### TCN Model

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=tcn_ctc \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

## Testing