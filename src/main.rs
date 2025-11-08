use crate::{
    dataset::DatasetBuilder,
    model::SimpleLstm,
    parser::read_csv,
    preprocessor::{Node, Pipeline},
};
// use burn::backend::{Candle, candle::CandleDevice};
// use burn::backend::{LibTorch, libtorch::LibTorchDevice};
// use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::backend::Autodiff;
use burn::backend::{NdArray, ndarray::NdArrayDevice};
use clap::{Parser, Subcommand};
use std::collections::HashMap;

mod batcher;
mod dataset;
mod model;
mod parser;
mod preprocessor;
mod train;

const PATHS: &[&str] = &[
    "data/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "data/PRSA_Data_Changping_20130301-20170228.csv",
    "data/PRSA_Data_Dingling_20130301-20170228.csv",
    "data/PRSA_Data_Dongsi_20130301-20170228.csv",
    "data/PRSA_Data_Guanyuan_20130301-20170228.csv",
];

const VALID_COLUMNS: &[&str] = &[
    "day", "hour", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd",
    "WSPM",
];

/// Parse a single key-value pair with pipeline
/// Format: output_name=source_column:TRANSFORM1,TRANSFORM2,...
/// If no colon, source_column defaults to output_name
fn parse_feature_pipeline(
    s: &str,
) -> Result<(String, String, Pipeline), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    let output_name = s[..pos].to_string();
    let rest = &s[pos + 1..];

    // Check if source column is specified with colon
    let (source_column, pipeline_str) = if let Some(colon_pos) = rest.find(':') {
        (rest[..colon_pos].to_string(), &rest[colon_pos + 1..])
    } else {
        // If no colon, source column = output name (backwards compatible)
        (output_name.clone(), rest)
    };

    // Validate source column name
    if !VALID_COLUMNS.contains(&source_column.as_str()) {
        return Err(format!(
            "Invalid source column: '{}'. Valid columns are: {}",
            source_column,
            VALID_COLUMNS.join(", ")
        )
        .into());
    }

    // Parse pipeline (left-to-right execution order)
    let nodes: Vec<Node> = if pipeline_str.trim().is_empty() {
        vec![]
    } else {
        pipeline_str
            .split_whitespace()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| Node::try_from(s))
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok((output_name, source_column, Pipeline::new(nodes)))
}

#[derive(Debug, Parser)]
#[command(
    name = "example-rnn-beijing-air-quality",
    version = "1.0",
    author = "Jakob",
    about = "Train RNN on Beijing air quality data"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train the model
    Train {
        // ============================================================
        // Model params
        // ============================================================
        /// Hidden layer size (hyperparameter to tune)
        #[arg(long, required = true)]
        hidden_size: usize,

        /// Learning rate
        #[arg(long, required = true)]
        learning_rate: f64,

        // ============================================================
        // Dataset & Preprocessing params
        // ============================================================
        /// Feature pipelines as key=value pairs (e.g., PM2.5=PM2.5:ROC(1),ZSCORE(100) or hour_sin=hour:SIN(24))
        #[clap(long = "feature", value_parser = parse_feature_pipeline)]
        features: Vec<(String, String, Pipeline)>,

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
    },

    /// Export preprocessed dataset to CSV
    Export {
        /// Feature pipelines as key=value pairs (e.g., PM2.5=PM2.5:ROC(1),ZSCORE(100) or hour_sin=hour:SIN(24))
        #[clap(long = "feature", value_parser = parse_feature_pipeline)]
        features: Vec<(String, String, Pipeline)>,

        /// Output CSV file path
        #[arg(long, required = true)]
        output: String,
    },
}

fn build_dataset(features: Vec<(String, String, Pipeline)>) -> anyhow::Result<DatasetBuilder> {
    let feature_length = features.len();
    let mut pipelines = HashMap::with_capacity(feature_length);
    let mut output_names = Vec::with_capacity(feature_length);
    let mut source_columns = Vec::with_capacity(feature_length);

    for (output_name, source_column, pipeline) in features {
        pipelines.insert(output_name.clone(), pipeline);
        output_names.push(output_name);
        source_columns.push(source_column);
    }

    let mut dataset_builder =
        DatasetBuilder::new(pipelines, output_names, source_columns.clone(), Some(35066));

    // Read and push all rows
    for result in read_csv(PATHS[0])? {
        let row = result?;

        // Create a record with ALL the source columns (not output names)
        let mut record = HashMap::new();
        for source_column in &source_columns {
            let value = match source_column.as_str() {
                "PM2.5" => row.pm2_5,
                "PM10" => row.pm10,
                "SO2" => row.so2,
                "NO2" => row.no2,
                "CO" => row.co,
                "O3" => row.o3,
                "TEMP" => row.temp,
                "PRES" => row.pres,
                "DEWP" => row.dewp,
                "RAIN" => row.rain,
                "WSPM" => row.wspm,
                "hour" => Some(row.hour as f32),
                "day" => Some(row.day as f32),
                "wd" => None, // String field, handle separately if needed
                _ => None,
            };

            if let Some(v) = value {
                record.insert(source_column.clone(), v);
            }
        }

        // Push to builder (skips rows where pipeline returns None)
        dataset_builder.push(record)?;
    }

    Ok(dataset_builder)
}

fn main() -> anyhow::Result<()> {
    type Backend = Autodiff<NdArray>;
    let device = NdArrayDevice::default();

    // type Backend = Autodiff<LibTorch>;
    // let device = LibTorchDevice::Cpu;

    // type Backend = Autodiff<Wgpu>;
    // let device = WgpuDevice::default();

    // type Backend = Autodiff<Candle>;
    // let device = CandleDevice::Cpu;

    let args = Args::parse();

    match args.command {
        Command::Train {
            hidden_size,
            learning_rate,
            features,
            sequence_length,
            prediction_horizon,
            batch_size,
            epochs,
        } => {
            let feature_length = features.len();
            let dataset_builder = build_dataset(features)?;

            let (dataset_training, dataset_validation) =
                dataset_builder.build(sequence_length, prediction_horizon, 0.8)?;

            let model =
                SimpleLstm::<Backend>::new(&device, feature_length, hidden_size, feature_length);

            // Train
            train::train(
                &device,
                &dataset_training,
                &dataset_validation,
                epochs,
                batch_size,
                learning_rate,
                model,
            );
        }

        Command::Export { features, output } => {
            let dataset_builder = build_dataset(features)?;
            dataset_builder.to_csv(&output)?;
            println!("Preprocessed dataset exported to: {}", output);
        }
    }

    Ok(())
}
