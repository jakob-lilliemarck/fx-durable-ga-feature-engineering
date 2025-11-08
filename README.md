## Installation
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 numpy==1.26.4 setuptools

# Make sure to set environment variables for libtorch, see .envrc
cargo install --path .
```

## Model training example
```sh
cargo run train \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --sequence-length 24 \
    --prediction-horizon 1 \
    --batch-size 64 \
    --epochs 50 \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)"

feng export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --output test.csv
```
