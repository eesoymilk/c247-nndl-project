# Commands

> This document contains the commands to train and test the models.

## Baseline TDS model

### Training

```bash
python -m emg2qwerty.train \
    user=single_user \
    model=tds_conv_ctc \
    checkpoint="result_logs/baseline-training/epoch73_step17760.ckpt" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    --multirun
```

### Testing

```bash
python -m emg2qwerty.train \
    user=single_user \
    model=tds_conv_ctc \
    checkpoint="result_logs/baseline_200-training/epoch98_step23760.ckpt" \
    train=False \
    trainer.accelerator=gpu \
    decoder=ctc_greedy \
    hydra.launcher.mem_gb=64 \
    --multirun
```

## Baseline TDS Model with Data Preprocessing

### Training

```bash
python -m emg2qwerty.train \
    user=single_user \
    model=tds_conv_ctc \
    transforms=custom \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    --multirun
```

### Testing

```bash
python -m emg2qwerty.train \
    user=single_user \
    model=tds_conv_ctc \
    checkpoint="result_logs/baseline_custom_transforms-training/epoch37_step9120.ckpt" \
    transforms=custom \
    train=False \
    trainer.accelerator=gpu \
    decoder=ctc_greedy \
    hydra.launcher.mem_gb=64 \
    --multirun
```

## TCN Model

### Training

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=tcn_ctc \
  checkpoint="result_logs/tcn-training/epoch99_step24000.ckpt" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

### Testing

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=tcn_ctc \
  checkpoint="result_logs/tcn_200-training/epoch0_step240.ckpt" \
  train=False \
  trainer.accelerator=gpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

## LSTM + GRU Model

### Training

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=lstm_gru_ctc \
  checkpoint="result_logs/lstm_gru-training/epoch95_step23040.ckpt" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

### Testing

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=lstm_gru_ctc \
  checkpoint="result_logs/lstm_gru_200-training/epoch93_step22560.ckpt" \
  train=False \
  trainer.accelerator=gpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

## Hybrid Model

> This model is a combination of TCN and LSTM + GRU models.

### Training

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=hybrid_ctc \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

### Testing

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=hybrid_ctc \
  checkpoint="result_logs/hybric_200-training/epoch99_step24000.ckpt" \
  train=False \
  trainer.accelerator=gpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```
