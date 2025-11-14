//! Inference evaluation over historical datasets.
//!
//! ## Performance Note: Batched Inference Opportunity
//!
//! Currently, this implementation processes rows sequentially:
//! 1. Read CSV row by row
//! 2. Apply pipelines to each row
//! 3. Make individual predictions (35k+ forward passes)
//!
//! For large datasets (35k rows), this can be slow (~1000+ forward passes).
//! Since we're operating on historical data, we could optimize by:
//!
//! **Proposed refactor:**
//! 1. First pass: Read all CSV rows, apply all pipelines → collect all_features/all_targets
//! 2. Second pass: Create all sequences at once (use SequenceDataset from dataset.rs)
//! 3. Third pass: Batch inference using SequenceBatcher (32 sequences/batch → ~1000 total batches)
//! 4. Match predictions back to row numbers
//!
//! Expected speedup: ~35x (from 35k single forward passes → 1k batched forward passes).
//! Tradeoff: Requires loading all preprocessed data into memory.

use crate::inference::InferenceEngine;
use crate::parser::read_csv;
use crate::preprocessor::Pipeline;
use burn::prelude::Backend;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Write;
use tracing::debug;

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub row_number: u32,                 // Actual row number from CSV file
    pub target: Vec<f32>,                // The actual value at the prediction target timestep
    pub prediction: Vec<f32>,            // Model prediction
    pub prediction_naive: Vec<f32>,      // Naive prediction using the value of the current timestep
    pub dist_prediction: Vec<f32>,       // |target - prediction|
    pub dist_prediction_naive: Vec<f32>, // |target - prediction_naive|
    pub baseline_accuracy: Vec<f32>, // dist_prediction_naive / dist_prediction if prediction is better, else 0
}

impl fmt::Display for InferenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] pred: {:?}, actual: {:?}, current: {:?}",
            self.row_number,
            format_vec(&self.prediction),
            format_vec(&self.target),
            format_vec(&self.prediction_naive)
        )
    }
}

fn format_vec(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>()
        .join(", ")
}

pub struct InferenceMetrics {
    pub mse: f32,
    pub rmse: f32,
    pub count: usize,
    pub mean_accuracy: f32,
    pub median_accuracy: f32,
    pub pct_above_baseline: f32, // % predictions > 0%
    pub pct_good: f32,           // % predictions > 50%
}

impl fmt::Display for InferenceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Predictions: {}, MSE: {:.6}, RMSE: {:.6}, Mean: {:.2}%, Median: {:.2}%, >Baseline: {:.1}%, >50%: {:.1}%",
            self.count,
            self.mse,
            self.rmse,
            self.mean_accuracy,
            self.median_accuracy,
            self.pct_above_baseline,
            self.pct_good
        )
    }
}

/// Run inference on a dataset, processing it sequentially
///
/// If `limit` is specified, stops after generating that many predictions.
/// If `limit` is None, processes the entire dataset.
pub fn run_inference<B: Backend>(
    engine: &InferenceEngine<B>,
    dataset_path: &str,
    limit: Option<usize>,
) -> anyhow::Result<Vec<InferenceResult>> {
    // Parse feature and target definitions from engine.config
    let features = parse_definitions(&engine.config.features)?;
    let targets = parse_definitions(&engine.config.targets)?;

    let sequence_length = engine.config.sequence_length;
    let prediction_horizon = engine.config.prediction_horizon;

    // Create fresh pipelines for inference
    let mut feature_pipelines: HashMap<String, Pipeline> = HashMap::with_capacity(features.len());
    let mut target_pipelines: HashMap<String, Pipeline> = HashMap::with_capacity(targets.len());

    for (output_name, _, pipeline) in &features {
        feature_pipelines.insert(output_name.clone(), pipeline.clone());
    }

    for (output_name, _, pipeline) in &targets {
        target_pipelines.insert(output_name.clone(), pipeline.clone());
    }

    // State tracking
    let mut feature_caches: HashMap<String, f32> = HashMap::new();
    let mut target_caches: HashMap<String, f32> = HashMap::new();

    // All timesteps collected so far (for warmup and buffering)
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_targets: Vec<Vec<f32>> = Vec::new();
    let mut all_row_numbers: Vec<u32> = Vec::new();
    let mut results = Vec::new();

    // Iterate through CSV rows
    for result in read_csv(dataset_path)? {
        let row = result?;
        let row_number = row.no;

        // Extract all source columns from row
        let mut record = HashMap::new();

        let wd_val: Option<f32> = row.wd.clone().map(|wd| wd.into());
        let all_source_cols = vec![
            ("PM2.5", row.pm2_5),
            ("PM10", row.pm10),
            ("SO2", row.so2),
            ("NO2", row.no2),
            ("CO", row.co),
            ("O3", row.o3),
            ("TEMP", row.temp),
            ("PRES", row.pres),
            ("DEWP", row.dewp),
            ("RAIN", row.rain),
            ("WSPM", row.wspm),
        ];

        for (col_name, value) in all_source_cols {
            if let Some(v) = value {
                record.insert(col_name.to_string(), v);
            }
        }

        record.insert("hour".to_string(), row.hour as f32);
        record.insert("day".to_string(), row.day as f32);
        record.insert("month".to_string(), row.month as f32);

        if let Some(wd) = wd_val {
            record.insert("wd".to_string(), wd);
        }

        // Process features through pipelines (matching dataset.rs pattern)
        // Process ALL features for this row before deciding to skip
        let mut feature_timestep = Vec::with_capacity(features.len());
        let mut skip_row = false;

        for (output_name, source_column, _) in &features {
            let pipeline = feature_pipelines.get_mut(output_name).unwrap();

            match record.get(source_column) {
                Some(value) => {
                    if let Some(processed) = pipeline.process(*value) {
                        feature_caches.insert(output_name.clone(), processed);
                        feature_timestep.push(processed);
                    } else {
                        skip_row = true;
                    }
                }
                None => {
                    if let Some(cached) = feature_caches.get(output_name) {
                        feature_timestep.push(*cached);
                    } else {
                        skip_row = true;
                    }
                }
            };
        }

        debug!(message="Features processed", row = ?row_number, feature_timestep = ?feature_timestep, skip_row=skip_row);

        if skip_row {
            continue;
        }

        // Process targets through pipelines
        // Process ALL targets for this row before deciding to skip
        let mut target_timestep = Vec::with_capacity(targets.len());

        for (output_name, source_column, _) in &targets {
            let pipeline = target_pipelines.get_mut(output_name).unwrap();

            match record.get(source_column) {
                Some(value) => {
                    if let Some(processed) = pipeline.process(*value) {
                        target_caches.insert(output_name.clone(), processed);
                        target_timestep.push(processed);
                    } else {
                        skip_row = true;
                    }
                }
                None => {
                    if let Some(cached) = target_caches.get(output_name) {
                        target_timestep.push(*cached);
                    } else {
                        skip_row = true;
                    }
                }
            };

            debug!(message="Target processed", row = row_number, target = %output_name, target_timestep=?target_timestep, skip_row=skip_row);
        }

        if skip_row {
            continue;
        }

        // Add to our buffers
        all_features.push(feature_timestep);
        all_targets.push(target_timestep);
        all_row_numbers.push(row_number);

        // Check if we have enough data to make a prediction
        if all_features.len() > sequence_length + prediction_horizon {
            let target_idx = all_features.len() - 1 - prediction_horizon;

            if target_idx >= sequence_length {
                // Extract sequence
                let sequence_start = target_idx - sequence_length;
                let sequence = all_features[sequence_start..target_idx].to_vec();

                // Get target value at prediction horizon
                let prediction_idx = target_idx + prediction_horizon;
                if prediction_idx < all_targets.len() {
                    let target = all_targets[prediction_idx].clone();
                    let prediction_naive = all_targets[target_idx].clone();
                    let pred_row_number = all_row_numbers[target_idx];

                    // Make prediction
                    match engine.predict(sequence) {
                        Ok(prediction) => {
                            // Calculate distance metrics
                            let dist_baseline: Vec<f32> = target
                                .iter()
                                .zip(prediction_naive.iter())
                                .map(|(a, c)| (a - c).abs())
                                .collect();

                            let dist_prediction: Vec<f32> = target
                                .iter()
                                .zip(prediction.iter())
                                .map(|(a, p)| (a - p).abs())
                                .collect();

                            let accuracy: Vec<f32> = dist_prediction
                                .iter()
                                .zip(dist_baseline.iter())
                                .map(|(dist_pred, dist_base)| {
                                    if dist_pred > dist_base {
                                        0.0
                                    } else if dist_base == &0.0 {
                                        if dist_pred == &0.0 { 100.0 } else { 0.0 }
                                    } else {
                                        (1.0 - (dist_pred / dist_base)) * 100.0
                                    }
                                })
                                .collect();

                            let result = InferenceResult {
                                row_number: pred_row_number,
                                prediction,
                                target,
                                prediction_naive,
                                dist_prediction_naive: dist_baseline,
                                dist_prediction,
                                baseline_accuracy: accuracy,
                            };
                            results.push(result);

                            // Check if we've reached the limit
                            if let Some(max_predictions) = limit {
                                if results.len() >= max_predictions {
                                    return Ok(results);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: inference failed at timestep: {}", e);
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Parse feature/target definitions back into (name, column, pipeline)
fn parse_definitions(definitions: &[String]) -> anyhow::Result<Vec<(String, String, Pipeline)>> {
    let mut result = Vec::new();

    for def in definitions {
        let pos = def
            .find('=')
            .ok_or_else(|| anyhow::anyhow!("Invalid definition: no '=' found in '{}'", def))?;
        let output_name = def[..pos].to_string();
        let rest = &def[pos + 1..];

        let (source_column, pipeline_str) = if let Some(colon_pos) = rest.find(':') {
            (rest[..colon_pos].to_string(), &rest[colon_pos + 1..])
        } else {
            (rest.to_string(), "")
        };

        // Parse pipeline
        let nodes: Vec<crate::preprocessor::Node> = if pipeline_str.trim().is_empty() {
            vec![]
        } else {
            pipeline_str
                .split_whitespace()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| crate::preprocessor::Node::try_from(s))
                .collect::<Result<Vec<_>, _>>()?
        };

        let pipeline = Pipeline::new(nodes);
        result.push((output_name, source_column, pipeline));
    }

    Ok(result)
}

/// Write inference results to CSV file
pub fn write_results_to_csv(results: &[InferenceResult], path: &str) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(
        file,
        "row_number,prediction,actual_value,current_value,dist_baseline,dist_prediction,accuracy"
    )?;

    // Write data rows
    for result in results {
        let pred_str = result
            .prediction
            .first()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let actual_str = result
            .target
            .first()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let current_str = result
            .prediction_naive
            .first()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let dist_base_str = result
            .dist_prediction_naive
            .first()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let dist_pred_str = result
            .dist_prediction
            .first()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let acc_str = result
            .baseline_accuracy
            .first()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "N/A".to_string());

        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            result.row_number,
            pred_str,
            actual_str,
            current_str,
            dist_base_str,
            dist_pred_str,
            acc_str
        )?
    }

    Ok(())
}

/// Calculate MSE, RMSE and accuracy metrics from inference results
pub fn calculate_metrics(results: &[InferenceResult]) -> InferenceMetrics {
    if results.is_empty() {
        return InferenceMetrics {
            mse: 0.0,
            rmse: 0.0,
            count: 0,
            mean_accuracy: 0.0,
            median_accuracy: 0.0,
            pct_above_baseline: 0.0,
            pct_good: 0.0,
        };
    }

    let mut sum_squared_error = 0.0;
    let mut accuracies = Vec::new();

    for result in results {
        // Assume single-value predictions and targets for simplicity
        if !result.prediction.is_empty() && !result.target.is_empty() {
            let pred = result.prediction[0];
            let actual = result.target[0];
            let error = pred - actual;
            sum_squared_error += error * error;

            // Collect accuracy values (assuming single output)
            if !result.baseline_accuracy.is_empty() {
                accuracies.push(result.baseline_accuracy[0]);
            }
        }
    }

    let mse = sum_squared_error / results.len() as f32;
    let rmse = mse.sqrt();

    // Calculate accuracy metrics
    let mean_accuracy = if !accuracies.is_empty() {
        accuracies.iter().sum::<f32>() / accuracies.len() as f32
    } else {
        0.0
    };

    // Sort for median calculation
    accuracies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_accuracy = if !accuracies.is_empty() {
        if accuracies.len() % 2 == 0 {
            (accuracies[accuracies.len() / 2 - 1] + accuracies[accuracies.len() / 2]) / 2.0
        } else {
            accuracies[accuracies.len() / 2]
        }
    } else {
        0.0
    };

    // Count predictions above baseline (> 0%)
    let above_baseline = accuracies.iter().filter(|a| **a > 0.0).count();
    let pct_above_baseline = if !accuracies.is_empty() {
        (above_baseline as f32 / accuracies.len() as f32) * 100.0
    } else {
        0.0
    };

    // Count good predictions (> 50%)
    let good_predictions = accuracies.iter().filter(|a| **a > 50.0).count();
    let pct_good = if !accuracies.is_empty() {
        (good_predictions as f32 / accuracies.len() as f32) * 100.0
    } else {
        0.0
    };

    InferenceMetrics {
        mse,
        rmse,
        count: results.len(),
        mean_accuracy,
        median_accuracy,
        pct_above_baseline,
        pct_good,
    }
}
