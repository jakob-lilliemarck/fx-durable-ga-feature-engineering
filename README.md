## Installation
```sh
cargo install --path .
```

## Model training example
```sh
fx-durable-ga-example-feature-engineering \
    --input-size 16 \
    --hidden-size 32 \
    --output-size 16 \
    --learning-rate 0.001 \
    --ema-window 10 \
    --ema-alpha 0.01 \
    --zscore-window 10
```
