use crate::{
    dataset::DatasetBuilder,
    model::{FeedForward, SequenceModel},
    parser::read_csv,
    preprocessor::{Node, Pipeline},
};
use burn::backend::Autodiff;
use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::data::dataloader::Dataset;
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use tracing::info;

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
    "data/PRSA_Data_Gucheng_20130301-20170228.csv",
    "data/PRSA_Data_Huairou_20130301-20170228.csv",
    "data/PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "data/PRSA_Data_Shunyi_20130301-20170228.csv",
    "data/PRSA_Data_Tiantan_20130301-20170228.csv",
    "data/PRSA_Data_Wanliu_20130301-20170228.csv",
    "data/PRSA_Data_Wanshouxigong_20130301-20170228.csv",
];

const VALID_COLUMNS: &[&str] = &[
    "day", "hour", "month", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
    "RAIN", "wd", "WSPM",
];

/// Parse a single key-value pair with optional pipeline
/// Format: output_name=source_column:TRANSFORM1 TRANSFORM2 ...
/// Or:     output_name=source_column  (no preprocessing)
fn parse_feature_pipeline(
    s: &str,
) -> Result<(String, String, Pipeline), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    let output_name = s[..pos].to_string();
    let rest = &s[pos + 1..];

    // Check if preprocessing is specified with colon
    let (source_column, pipeline_str) = if let Some(colon_pos) = rest.find(':') {
        (rest[..colon_pos].to_string(), &rest[colon_pos + 1..])
    } else {
        // If no colon, no preprocessing - just use the source column directly
        (rest.to_string(), "")
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
        /// Feature pipelines as key=value pairs (e.g., temp_ema=TEMP:ZSCORE(100) or hour_sin=hour:SIN(24))
        #[clap(long = "feature", value_parser = parse_feature_pipeline)]
        features: Vec<(String, String, Pipeline)>,

        /// Target pipelines as key=value pairs (e.g., target_temp=TEMP or target_pm25=PM2.5:ZSCORE(100))
        #[clap(long = "target", value_parser = parse_feature_pipeline)]
        targets: Vec<(String, String, Pipeline)>,

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

fn build_dataset_from_file(
    path: &str,
    features: &[(String, String, Pipeline)],
    targets: &[(String, String, Pipeline)],
) -> anyhow::Result<DatasetBuilder> {
    // Process features (clone pipelines for each file)
    let mut feature_pipelines = HashMap::with_capacity(features.len());
    let mut feature_output_names = Vec::with_capacity(features.len());
    let mut feature_source_columns = Vec::with_capacity(features.len());

    for (output_name, source_column, pipeline) in features {
        feature_pipelines.insert(output_name.clone(), pipeline.clone());
        feature_output_names.push(output_name.clone());
        feature_source_columns.push(source_column.clone());
    }

    // Process targets (clone pipelines for each file)
    let mut target_pipelines = HashMap::with_capacity(targets.len());
    let mut target_output_names = Vec::with_capacity(targets.len());
    let mut target_source_columns = Vec::with_capacity(targets.len());

    for (output_name, source_column, pipeline) in targets {
        target_pipelines.insert(output_name.clone(), pipeline.clone());
        target_output_names.push(output_name.clone());
        target_source_columns.push(source_column.clone());
    }

    // Collect all unique source columns needed
    let mut all_source_columns = feature_source_columns.clone();
    for col in &target_source_columns {
        if !all_source_columns.contains(col) {
            all_source_columns.push(col.clone());
        }
    }

    let mut dataset_builder = DatasetBuilder::new(
        feature_pipelines,
        feature_output_names,
        feature_source_columns,
        target_pipelines,
        target_output_names,
        target_source_columns,
        Some(35066),
    );

    // Read and push all rows from specified file
    for result in read_csv(path)? {
        let row = result?;

        // Create a record with ALL the source columns needed
        let mut record = HashMap::new();
        for source_column in &all_source_columns {
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
                "month" => Some(row.month as f32),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_feature_with_preprocessing() {
        let result = parse_feature_pipeline("output=TEMP:ZSCORE(10)");
        assert!(result.is_ok());
        let (output_name, source_column, pipeline) = result.unwrap();
        assert_eq!(output_name, "output");
        assert_eq!(source_column, "TEMP");
        assert_eq!(pipeline.nodes.len(), 1);
    }

    #[test]
    fn test_parse_feature_without_preprocessing() {
        let result = parse_feature_pipeline("target_temp=TEMP");
        assert!(result.is_ok());
        let (output_name, source_column, pipeline) = result.unwrap();
        assert_eq!(output_name, "target_temp");
        assert_eq!(source_column, "TEMP");
        assert_eq!(pipeline.nodes.len(), 0); // No preprocessing
    }

    #[test]
    fn test_parse_feature_invalid_column() {
        let result = parse_feature_pipeline("output=INVALID_COL");
        assert!(result.is_err());
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .pretty()
        .init();

    type Backend = Autodiff<NdArray>;
    let device = NdArrayDevice::default();

    let args = Args::parse();

    match args.command {
        Command::Train {
            hidden_size,
            learning_rate,
            features,
            targets,
            sequence_length,
            prediction_horizon,
            batch_size,
            epochs,
        } => {
            let feature_length = features.len();
            let target_length = targets.len();

            // Load data from all weather stations and combine their sequences
            let mut all_train_items = Vec::new();
            let mut all_valid_items = Vec::new();

            for (i, path) in PATHS.iter().enumerate() {
                info!(
                    message = format!("Loading file [{}/{}]", i + 1, PATHS.len()),
                    station = path,
                    path
                );

                let dataset_builder = build_dataset_from_file(path, &features, &targets)?;
                let (train, valid) =
                    dataset_builder.build(sequence_length, prediction_horizon, 0.8)?;

                let train_len = train.len();
                let valid_len = valid.len();

                // Collect items from this station
                for idx in 0..train_len {
                    all_train_items.push(train.get(idx).unwrap());
                }
                for idx in 0..valid_len {
                    all_valid_items.push(valid.get(idx).unwrap());
                }
            }

            info!(
                message = "Loading and preprocessing completed",
                training_sequences = all_train_items.len(),
                validation_sequences = all_valid_items.len()
            );

            // Create combined datasets
            let dataset_training = dataset::SequenceDataset::from_items(all_train_items);
            let dataset_validation = dataset::SequenceDataset::from_items(all_valid_items);

            // Choose model architecture:
            let model = FeedForward::<Backend>::new(
                &device,
                feature_length,
                hidden_size,
                target_length,
                sequence_length,
            );
            // let model = SimpleRnn::<Backend>::new(&device, feature_length, hidden_size, target_length, sequence_length);
            // let model = SimpleLstm::<Backend>::new(&device, feature_length, hidden_size, target_length, sequence_length);

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
            // For export, use features as both features and targets from first station
            let dataset_builder = build_dataset_from_file(PATHS[0], &features, &features)?;
            dataset_builder.to_csv(&output)?;
            println!("Preprocessed dataset exported to: {}", output);
        }
    }

    Ok(())
}
