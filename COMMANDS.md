## Training

### Baseline TDS model

```bash
python -m emg2qwerty.train \
  user="single_user" \
  model="tds_conv_ctc" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

### Baseline TDS Model with Data Preprocessing

```bash
python -m emg2qwerty.train \
  user="single_user" \
  model="tds_conv_ctc" \
  transforms="custom" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

### TCN Model

```bash
python -m emg2qwerty.train \
  user="single_user" \
  model="tcn_ctc" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  --multirun
```

## Testing