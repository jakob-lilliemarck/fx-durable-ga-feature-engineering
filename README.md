# fx-durable-ga-feature-engineering
A tiny CLI tool to train a neural net on the Beijing Multi-Site Air Quality dataset, providing a syntax and functionality to explore a wide range of feature and preprocessing strategies. The purpose of this repository and Rust crate is to serve as an example of how to use `fx-durable-ga` to perform automated feature-engineering over a constrained, but very vast search space.

*Dataset*: Beijing Multi-Site Air Quality by Song Chen (UCI ML Repository, CC BY 4.0) - https://doi.org/10.24432/C5RK5G

## Model training example
This example was the best performer I was able to find manually after playing with it for about 2 hours. Spoiler - the GA surpassed that on the first attempt in a much shorter timeframe.
```sh
cargo run --release -- train \
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
    --target "target_temp=TEMP" \
    --model-save-path ./my_model
```

### Saving the trained model
To save the trained model to a file, use the `--model-save-path` argument:
```sh
cargo run --release -- train \
    --hidden-size 16 \
    --learning-rate 0.01 \
    --sequence-length 25 \
    --prediction-horizon 1 \
    --batch-size 50 \
    --epochs 10 \
    --feature "temp_zscore=TEMP:ZSCORE(48)" \
    --target "target_temp=TEMP" \
    --model-save-path ./my_model
```

The model will be saved using Burn's `CompactRecorder` format. If `--model-save-path` is not specified, the model is not saved (useful when running under the GA system).

### Running inference on unseen data
Once a model is trained and saved, you can run inference on the unseen Wanshouxigong dataset:

```sh
feng infer --model-path ./my_model
```

The inference command will:
1. Load the saved model and its configuration from `./my_model` and `./my_model.config.json`
2. Process the Wanshouxigong dataset sequentially using the same preprocessing pipelines that were used during training
3. Generate predictions for each timestep
4. Output the number of timesteps processed

Important: The model configuration (features, targets, sequence length, prediction horizon, etc.) is automatically loaded from the config file, so you only need to specify the model path.

### Exporting preprocessed data
To export pre-processed data to csv, in order to plot or validate the preprocessed data, use the `export` command
```sh
feng export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --output test.csv
```
