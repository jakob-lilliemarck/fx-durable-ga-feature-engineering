# fx-durable-ga-feature-engineering
A tiny CLI tool to train a neural net on the Beijing Multi-Site Air Quality dataset, providing a syntax and functionality to explore a wide range of feature and preprocessing strategies. The purpose of this repository and Rust crate is to serve as an example of how to use `fx-durable-ga` to perform automated feature-engineering over a constrained, but very vast search space.

*Dataset*: Beijing Multi-Site Air Quality by Song Chen (UCI ML Repository, CC BY 4.0) - https://doi.org/10.24432/C5RK5G

## Model training example
This example was the best performer I was able to find manually after playing with it for about 2 hours. Spoiler - the GA surpassed that on the first attempt in a much shorter timeframe.
```sh
cargo run -- train \
    --hidden-size 64 \
    --learning-rate 0.0001 \
    --sequence-length 48 \
    --prediction-horizon 1 \
    --batch-size 100 \
    --epochs 25 \
    --feature "temp_zscore=TEMP:ZSCORE(48)" \
    --feature "temp=TEMP:ZSCORE(48)" \
    --feature "temp_roc_1=TEMP:ROC(1) ZSCORE(48)" \
    --feature "temp_roc_4=TEMP:ROC(4) ZSCORE(48)" \
    --feature "pres=PRES:ZSCORE(48)" \
    --feature "pres_roc_1=PRES:ROC(1) ZSCORE(48)" \
    --feature "pres_roc_4=PRES:ROC(4) ZSCORE(48)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --feature "month_sin=month:SIN(12)" \
    --feature "month_cos=month:COS(12)" \
    --target "target_temp=TEMP"
```

To export pre-processed data to csv, in order to plot or validate the preprocessed data, use the `export` command
```sh
feng export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --output test.csv
```
