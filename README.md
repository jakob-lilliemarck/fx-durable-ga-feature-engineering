## Installation
```sh
cargo install --path .
```

## Model training example
```sh
train train \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --sequence-length 100 \
    --prediction-horizon 1 \
    --batch-size 8 \
    --epochs 100 \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)"

train export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --output test.csv
```
