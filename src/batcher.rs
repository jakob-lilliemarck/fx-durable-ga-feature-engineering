use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use super::dataset::SequenceDatasetItem;

#[derive(Clone, Debug)]
pub struct SequenceBatch<B: Backend> {
    pub sequences: Tensor<B, 3>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct SequenceBatcher<B: Backend> {
    // you can still store device here if you want,
    // but it's not required since it's passed into batch()
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> SequenceBatcher<B> {
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, SequenceDatasetItem, SequenceBatch<B>> for SequenceBatcher<B> {
    fn batch(&self, items: Vec<SequenceDatasetItem>, device: &B::Device) -> SequenceBatch<B> {
        let batch_size = items.len();
        assert!(batch_size > 0, "Cannot create a batch from an empty Vec");

        let seq_len = items[0].sequence.len();
        let feature_dim = 8;

        // Flatten everything
        let mut all_sequences = Vec::with_capacity(batch_size * seq_len * feature_dim);
        let mut all_targets = Vec::with_capacity(batch_size * feature_dim);

        for item in items.iter() {
            for timestep in item.sequence.iter() {
                all_sequences.extend_from_slice(timestep);
            }
            all_targets.extend_from_slice(&item.target);
        }

        // Create tensors
        let sequences = Tensor::<B, 3>::from_floats(all_sequences.as_slice(), device).reshape([
            batch_size,
            seq_len,
            feature_dim,
        ]);

        let targets = Tensor::<B, 2>::from_floats(all_targets.as_slice(), device)
            .reshape([batch_size, feature_dim]);

        SequenceBatch { sequences, targets }
    }
}
