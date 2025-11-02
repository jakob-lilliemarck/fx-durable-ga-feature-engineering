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
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_training_converges() {
        // Autodiff with NoCheckpointing uses GradientsParams
        type Backend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        // Simple repeating pattern: 0, 1, 2, 3, 2, 1, 0, 1, ...
        let pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0];

        let mut dataset_train = SequenceDataset::new(3, 1);
        dataset_train.new_batch();
        for i in 0..100 {
            let value = pattern[i % pattern.len()];
            dataset_train.push([value; 16]);
        }
        dataset_train.finalize();

        let mut dataset_valid = SequenceDataset::new(3, 1);
        dataset_valid.new_batch();
        for i in 0..100 {
            let value = pattern[i % pattern.len()];
            dataset_valid.push([value; 16]);
        }
        dataset_valid.finalize();

        // Create model
        let model = SimpleRnn::<Backend>::new(&device, 16, 32, 16);

        // Train
        train(
            &device,
            &dataset_train,
            &dataset_valid,
            25,
            16,
            0.001,
            model,
        );

        println!("Training complete! Check if losses decreased.");
    }
}
