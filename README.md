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
cargo run -- train \
    --hidden-size 64 \
    --learning-rate 0.0003 \
    --sequence-length 24 \
    --prediction-horizon 1 \
    --batch-size 64 \
    --epochs 100 \
    --feature "temp_zscore=TEMP:ZSCORE(48)" \
    --feature "temp=TEMP:ZSCORE(48)" \
    --feature "temp_roc_1=TEMP:ROC(1) ZSCORE(48)" \
    --feature "temp_roc_4=TEMP:ROC(4) ZSCORE(48)" \
    --feature "pres=PRES:ZSCORE(48)" \
    --feature "pres_roc_1=PRES:ROC(1) ZSCORE(48)" \
    --feature "pres_roc_4=PRES:ROC(4) ZSCORE(48)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --target "target_temp=TEMP"

feng export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --output test.csv
```
