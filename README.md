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

feng export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)" \
    --output test.csv
```

## Targets

### Single Target (Recommended)

For simplicity and interpretability, use one target per model with **raw, unnormalized values**:

```sh
--target "target_temp=TEMP"
```

**Advantages:**
- Predictions are directly interpretable (no denormalization needed)
- No distribution shift issues between training and inference
- Simpler to reason about model performance
- Each model can be optimized independently for its specific task

**Note:** Loss values will be larger for raw targets (e.g., MSE ~80 for temperature in °C), but this is expected and doesn't affect model quality.

### Multiple Targets (Advanced)

The model supports predicting multiple targets simultaneously. The loss function (MSE) averages errors across all targets:

```rust
loss = mean(squared_errors_for_all_targets)
```

**Important considerations:**

1. **Scale imbalance:** Targets with different scales (e.g., temperature in °C and pressure in hPa) will cause the larger-scale target to dominate the loss. You must normalize targets to similar scales.

2. **ZSCORE normalization issues:** Rolling window ZSCORE (e.g., ZSCORE(48)) creates distribution shift problems:
   - Summer temps: mean=26°C, std=1.3 → z-score=+1.0 means "slightly warm"
   - Winter temps: mean=-2°C, std=2.1 → z-score=+1.0 means "very warm"
   - Model learns z-score patterns that don't transfer across seasons
   - Denormalization requires maintaining a rolling buffer of recent target values

**Recommendation:** Use separate single-target models instead of multi-target models unless you specifically need joint predictions and are willing to handle the complexity.
