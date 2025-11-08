use super::dataset::SequenceDatasetItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

#[derive(Clone, Debug)]
pub struct SequenceBatch<B: Backend> {
    pub sequences: Tensor<B, 3>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct SequenceBatcher<B: Backend> {
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
        let feature_dim = items[0].sequence[0].len();

        // Pre-allocate with exact capacity - this is the key optimization!
        // No reallocations = much faster and no memory fragmentation
        let total_seq_elements = batch_size * seq_len * feature_dim;
        let total_target_elements = batch_size * feature_dim;

        let mut all_sequences = Vec::with_capacity(total_seq_elements);
        let mut all_targets = Vec::with_capacity(total_target_elements);

        // Now extend_from_slice won't cause any reallocations
        for item in items.iter() {
            for timestep in item.sequence.iter() {
                all_sequences.extend_from_slice(timestep);
            }
            all_targets.extend_from_slice(&item.target);
        }

        // Create tensors directly from the Vec data
        let sequences = Tensor::<B, 3>::from_data(
            TensorData::new(all_sequences, [batch_size, seq_len, feature_dim]),
            device,
        );

        let targets = Tensor::<B, 2>::from_data(
            TensorData::new(all_targets, [batch_size, feature_dim]),
            device,
        );

        SequenceBatch { sequences, targets }
    }
}
