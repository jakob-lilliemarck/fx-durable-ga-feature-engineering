use crate::{
    dataset::DatasetBuilder,
    model::SimpleRnn,
    parser::read_csv,
    preprocessor::{Node, Pipeline},
};
use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};
use clap::Parser;
use std::collections::HashMap;

mod batcher;
mod dataset;
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
    // Dataset & Preprocessing params
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
    /// Sequence length
    #[arg(long, required = true)]
    sequence_length: usize,

    /// Prediction horizon
    #[arg(long, required = true)]
    prediction_horizon: usize,

    /// Batch size
    #[clap(long, default_value_t = 8)]
    batch_size: usize,

    /// Epochs
    #[clap(long, default_value_t = 100)]
    epochs: usize,
}

const PATHS: &[&str] = &[
    "data/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "data/PRSA_Data_Changping_20130301-20170228.csv",
    "data/PRSA_Data_Dingling_20130301-20170228.csv",
    "data/PRSA_Data_Dongsi_20130301-20170228.csv",
    "data/PRSA_Data_Guanyuan_20130301-20170228.csv",
];

fn main() -> anyhow::Result<()> {
    // Autodiff with NoCheckpointing uses GradientsParams
    type Backend = Autodiff<NdArray>;
    let device = NdArrayDevice::default();

    // --- Parse CLI arguments
    let args = Args::parse();

    let features = &["pm2_5"];
    let mut pipelines = HashMap::with_capacity(1);
    pipelines.insert(
        "pm2_5",
        Pipeline::new([
            Node::ema(args.ema_window, args.ema_alpha),
            Node::zscore(args.zscore_window),
        ]),
    );

    let mut dataset_builder = DatasetBuilder::new(pipelines, features, Some(35066));

    // Read and push all rows
    for result in read_csv(PATHS[0])? {
        let row = result?;

        // Create a record with the features we want
        let mut record = HashMap::new();
        if let Some(value) = row.pm2_5 {
            record.insert("pm2_5".to_string(), value);
        }

        // Push to builder (skips rows where pipeline returns None)
        dataset_builder.push(record)?;
    }

    let (dataset_training, dataset_validation) =
        dataset_builder.build(args.sequence_length, args.prediction_horizon, 0.8)?;

    let model =
        SimpleRnn::<Backend>::new(&device, args.input_size, args.hidden_size, args.output_size);

    // Train
    train::train(
        &device,
        &dataset_training,
        &dataset_validation,
        args.epochs,
        8,
        args.learning_rate,
        model,
    );

    Ok(())
}
