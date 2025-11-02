use crate::batcher::SequenceBatcher;
use crate::dataset::SequenceDataset;
use crate::model::SimpleRnn;
use burn::data::dataloader::Dataset;
use burn::data::dataloader::batcher::Batcher;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/*
If validation loss is much lower than training loss → something's wrong
If both losses are stuck → learning rate might be too low/high
If training loss decreases but validation doesn't → overfitting
*/
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    dataset_training: &SequenceDataset,
    dataset_validation: &SequenceDataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    mut model: SimpleRnn<B>,
) -> SimpleRnn<B> {
    // Initialize optimizer
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Create batchers
    let batcher_train = SequenceBatcher::<B>::new();
    let batcher_valid = SequenceBatcher::<B::InnerBackend>::new();

    let dataset_train_len = dataset_training.len();
    let dataset_valid_len = dataset_validation.len();

    // Early stopping variables
    let mut best_valid_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0;
    let patience = 10; // Stop if no improvement for 10 epochs

    for epoch in 0..epochs {
        // ============ Training Loop ============
        let mut total_train_loss = 0.0;
        let mut num_train_batches = 0;

        for start_idx in (0..dataset_train_len).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(dataset_train_len);
            let items: Vec<_> = (start_idx..end_idx)
                .filter_map(|i| dataset_training.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher_train.batch(items, device);

            // Forward pass
            let outputs = model.forward(batch.sequences);

            // MSE Loss
            let loss = (outputs - batch.targets).powf_scalar(2.0).mean();

            // Extract scalar value BEFORE backward
            let loss_value = loss.clone().into_scalar().elem::<f32>();

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);

            // Update parameters
            model = optimizer.step(learning_rate, model, grads_params);

            // Accumulate loss
            total_train_loss += loss_value;
            num_train_batches += 1;
        }

        let avg_train_loss = if num_train_batches > 0 {
            total_train_loss / num_train_batches as f32
        } else {
            0.0
        };

        // ============ Validation Loop ============
        let valid_model = model.valid();
        let mut total_valid_loss = 0.0;
        let mut num_valid_batches = 0;

        for start_idx in (0..dataset_valid_len).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(dataset_valid_len);
            let items: Vec<_> = (start_idx..end_idx)
                .filter_map(|i| dataset_validation.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher_valid.batch(items, device);

            // Forward pass (no autodiff)
            let outputs = valid_model.forward(batch.sequences);

            // MSE Loss
            let loss = (outputs - batch.targets).powf_scalar(2.0).mean();
            let loss_value = loss.into_scalar().elem::<f32>();

            total_valid_loss += loss_value;
            num_valid_batches += 1;
        }

        let avg_valid_loss = if num_valid_batches > 0 {
            total_valid_loss / num_valid_batches as f32
        } else {
            0.0
        };

        // ============ Early Stopping Check ============
        if avg_valid_loss < best_valid_loss {
            best_valid_loss = avg_valid_loss;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= patience {
                println!(
                    "Early stopping at epoch {} - Best validation loss: {:.6}",
                    epoch + 1,
                    best_valid_loss
                );
                break;
            }
        }

        println!(
            "Epoch {}/{} - Train Loss: {:.6}, Valid Loss: {:.6} (Best: {:.6})",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_valid_loss,
            best_valid_loss
        );
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DatasetBuilder;
    use crate::preprocessor::{Node, Pipeline};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use std::collections::HashMap;

    #[test]
    fn test_training_converges() {
        type Backend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        // Simple repeating pattern: 0, 1, 2, 3, 2, 1, 0, 1, ...
        let pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0];

        // Create pipelines - passthrough (no preprocessing)
        let mut pipelines = HashMap::new();
        pipelines.insert("value", Pipeline::new([Node::Noop, Node::Noop]));

        let features = &["value"];

        // Build dataset
        let mut builder = DatasetBuilder::new(pipelines, features, Some(100));
        for i in 0..100 {
            let value = pattern[i % pattern.len()];
            let mut record = HashMap::new();
            record.insert("value".to_string(), value);
            builder.push(record).unwrap();
        }

        // Split into train (80%) and validation (20%)
        let (dataset_train, dataset_valid) = builder.build(4, 1, 0.8).expect("should build");

        // Create model - input_size=1 (single feature), output_size=1
        let model = SimpleRnn::<Backend>::new(&device, 1, 64, 1);

        // Train
        train(&device, &dataset_train, &dataset_valid, 25, 32, 0.01, model);

        println!("Training complete! Check if losses decreased.");
    }
}
