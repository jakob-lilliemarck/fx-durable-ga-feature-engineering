# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Purpose

This is an **example project** demonstrating automated feature engineering using the `fx-durable-ga` genetic algorithm optimization system. The binary (`feng`) is designed to be invoked by the GA system as a subprocess for fitness evaluation.

**Why a separate process?** Burn's autodiff backend leaks memory during training - variables accumulate and aren't freed. Running each training session in a separate process ensures memory is actually released after each GA evaluation.

## Current State & Planned Changes

**Current setup:**
- Uses LibTorch backend (via Python venv) for ~15-20% speedup on CPU
- Uses LSTM model on Beijing air quality dataset
- Requires fragile Python environment setup

**Planned changes:**
- Switch back to **wgpu or ndarray backend** (simpler, no Python dependency)
- Move to **simpler model architecture** (LSTM is too complex/slow for GA optimization feasibility)
- The goal is to demonstrate GA optimization results, not showcase the model itself

## Environment Setup

**Current (with LibTorch - will be removed):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 numpy==1.26.4 setuptools
cargo install --path .
```

Environment variables in `.envrc` required for libtorch (temporary, will be removed when switching backends).

**Future (simplified):**
```bash
cargo install --path .
```

## Commands

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test
```

### Train Model (invoked by GA system)
```bash
feng train \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --sequence-length 100 \
    --prediction-horizon 1 \
    --batch-size 16 \
    --epochs 100 \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --feature "hour_sin=hour:SIN(24)" \
    --feature "hour_cos=hour:COS(24)"
```

The GA system optimizes by varying:
- Feature pipeline configurations (transforms, window sizes)
- Model hyperparameters (hidden size, learning rate)
- Training parameters (batch size, sequence length, etc.)

### Export Preprocessed Data
```bash
feng export \
    --feature "temp_ema=TEMP:ZSCORE(100)" \
    --output test.csv
```

## Architecture

### High-Level Flow

1. **GA System** → spawns `feng train` subprocess with specific parameters
2. **feng** → loads data, applies feature pipeline, trains model, returns fitness score
3. **Process exits** → memory is freed (workaround for Burn memory leak)
4. **GA System** → evaluates fitness, evolves population, repeat

### Feature Pipeline System

Features are defined via CLI: `output_name=source_column:TRANSFORM1 TRANSFORM2 ...`

**Available transforms** (in `src/preprocessor.rs`):
- `ZSCORE(window)` - Z-score normalization
- `EMA(window, alpha)` - Exponential moving average
- `STD(window)` - Rolling standard deviation  
- `ROC(offset)` - Rate of change
- `SIN(period)` / `COS(period)` - Cyclic encoding

Transforms are stateful and applied sequentially. Many require warmup period before producing output.

### Key Components

- **`src/parser.rs`** - Reads Beijing air quality CSV data
- **`src/preprocessor.rs`** - Transform implementations (EMA, ZSCORE, etc.)
- **`src/dataset.rs`** - Applies pipelines, creates sequence windows, handles train/val split
- **`src/batcher.rs`** - Batches sequences into tensors
- **`src/model.rs`** - Model architectures (currently LSTM, will simplify)
- **`src/train.rs`** - Training loop with early stopping
- **`src/main.rs`** - CLI interface and dataset building

### Data

Beijing air quality dataset in `data/` directory. Only uses `PRSA_Data_Aotizhongxin_20130301-20170228.csv` by default.

Valid columns: day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM

## Integration with fx-durable-ga

The GA system will:
1. Define parameter space (feature transforms, model params, training params)
2. Encode/decode parameters to/from genes
3. Spawn `feng train` with decoded parameters
4. Read fitness score (validation loss) from output
5. Evolve population based on fitness

This crate's job: accept parameters → train → return fitness. That's it.

## Backend Configuration

In `src/main.rs`, uncomment the desired backend:
- `LibTorch` (current, will be removed)
- `Wgpu` (preferred future option)
- `NdArray` (alternative future option)
- `Candle` (available but not tested)
