use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};
use clap::Parser;

use crate::{loader::load_and_preprocess, model::SimpleRnn};

mod batcher;
mod dataset;
mod loader;
mod model;
mod parser;
mod preprocessor;
mod train;

/// Train a RNN on "Gas sensor array draft dataset"
#[derive(Debug, Parser)]
#[command(
    name = "example-rnn-gas-sensor-array",
    version = "1.0",
    author = "Jakob",
    about = "Give credits"
)]
struct Args {
    // ============================================================
    // Model params
    // ============================================================
    /// Hidden layer size
    #[arg(long, required = true)]
    input_size: usize,

    /// Hidden layer size
    #[arg(long, required = true)]
    hidden_size: usize,

    /// Hidden layer size
    #[arg(long, required = true)]
    output_size: usize,

    /// Learning rate
    #[arg(long, required = true)]
    learning_rate: f64,

    // ============================================================
    // Preprocessing params
    // ============================================================
    /// Ema window size
    #[clap(long, required = true)]
    ema_window: usize,

    /// Ema alpha
    #[clap(long, required = true)]
    ema_alpha: f32,

    /// Zscore window size
    #[clap(long, required = true)]
    zscore_window: usize,

    // ============================================================
    // Training params
    // ============================================================
    /// Epochs
    #[clap(long, default_value_t = 100)]
    epochs: usize,
}

const PATHS: [&str; 10] = [
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch1.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch2.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch3.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch4.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch5.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch6.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch7.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch8.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch9.dat",
    "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch10.dat",
];
const SEQUENCE_LENGTH: usize = 10;
const PREDICTION_HORIZON: usize = 1;
const SENSOR_ARRAY_LENGTH: usize = 16;

fn main() -> anyhow::Result<()> {
    // Autodiff with NoCheckpointing uses GradientsParams
    type Backend = Autodiff<NdArray>;
    let device = NdArrayDevice::default();

    // --- Parse CLI arguments
    let args = Args::parse();

    let dataset_train = load_and_preprocess(
        &PATHS[0..4],
        SEQUENCE_LENGTH,
        PREDICTION_HORIZON,
        args.ema_window,
        args.ema_alpha,
        args.zscore_window,
    )?;

    let dataset_valid = load_and_preprocess(
        &PATHS[4..5],
        SEQUENCE_LENGTH,
        PREDICTION_HORIZON,
        args.ema_window,
        args.ema_alpha,
        args.zscore_window,
    )?;

    let model = SimpleRnn::<Backend>::new(&device, 16, 32, 16);

    // Train
    train::train(
        &device,
        &dataset_train,
        &dataset_valid,
        args.epochs,
        SENSOR_ARRAY_LENGTH,
        args.learning_rate,
        model,
    );

    Ok(())
}
