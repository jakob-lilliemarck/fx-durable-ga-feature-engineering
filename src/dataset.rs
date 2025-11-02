use burn::data::dataset::Dataset;
use tabled::Tabled;

pub const NUMBER_OF_SENSORS: usize = 16;
pub const NUMBER_OF_BATCHES: usize = 10;

pub type Timestep = [f32; NUMBER_OF_SENSORS];

pub struct SequenceDatasetItem {
    // [timesteps, sensors]
    pub sequence: Vec<Timestep>,
    pub target: Timestep,
}

pub struct Batch {
    idx: usize,
    features: Vec<Timestep>,
}

pub struct SequenceDataset {
    sequence_length: usize,
    prediction_horizon: usize,
    batches: Vec<Batch>,
    offsets: Vec<(usize, usize)>,
    sequences: usize,
}

impl SequenceDataset {
    pub fn new(sequence_length: usize, prediction_horizon: usize) -> Self {
        Self {
            sequence_length,
            prediction_horizon,
            batches: Vec::with_capacity(NUMBER_OF_BATCHES),
            offsets: Vec::with_capacity(NUMBER_OF_BATCHES),
            sequences: 0,
        }
    }

    pub fn new_batch(&mut self) {
        let cur = self.batches.len();

        let idx = cur + 1;

        self.batches.push(Batch {
            idx,
            features: Vec::new(),
        })
    }

    /// Push a timestep to the current batch
    pub fn push(&mut self, record: Timestep) {
        if let Some(current_batch) = self.batches.last_mut() {
            current_batch.features.push(record);
        } else {
            panic!("Must call new_batch() before push()");
        }
    }

    /// Call this after all data is loaded to build the index
    pub fn finalize(&mut self) {
        self.offsets.clear();
        self.sequences = 0;

        for (batch_idx, batch) in self.batches.iter().enumerate() {
            let batch_len = batch.features.len();
            let min_required = self.sequence_length + self.prediction_horizon;

            if batch_len < min_required {
                eprintln!(
                    "Warning: Batch {} has {} timesteps but needs {} (skipping)",
                    batch.idx, batch_len, min_required
                );
                continue;
            }

            // Calculate how many valid sequences this batch can produce
            let num_sequences = batch_len - self.sequence_length - self.prediction_horizon + 1;

            for local_idx in 0..num_sequences {
                self.offsets.push((batch_idx, local_idx));
            }

            self.sequences += num_sequences;
        }

        println!(
            "Dataset finalized: {} batches, {} total sequences",
            self.batches.len(),
            self.sequences
        );
    }

    pub fn get_item(&self, index: usize) -> Option<SequenceDatasetItem> {
        if index >= self.offsets.len() {
            return None;
        }

        let (batch_idx, local_idx) = self.offsets[index];
        let batch = &self.batches[batch_idx];

        let target_idx = local_idx + self.sequence_length + self.prediction_horizon - 1;

        if target_idx >= batch.features.len() {
            return None;
        }

        let sequence = batch.features[local_idx..local_idx + self.sequence_length].to_vec();
        let target = batch.features[target_idx];

        Some(SequenceDatasetItem { sequence, target })
    }

    /// Get statistics about the dataset
    pub fn stats(&self) -> DatasetStats {
        DatasetStats {
            num_batches: self.batches.len(),
            total_sequences: self.sequences,
            sequence_length: self.sequence_length,
            prediction_horizon: self.prediction_horizon,
            batches: self
                .batches
                .iter()
                .map(|b| BatchStats {
                    batch_idx: b.idx,
                    num_timesteps: b.features.len(),
                    num_sequences: if b.features.len()
                        >= self.sequence_length + self.prediction_horizon
                    {
                        b.features.len() - self.sequence_length - self.prediction_horizon + 1
                    } else {
                        0
                    },
                })
                .collect(),
        }
    }
}

#[derive(Tabled)]
pub struct DatasetStats {
    pub num_batches: usize,
    pub total_sequences: usize,
    pub sequence_length: usize,
    pub prediction_horizon: usize,
    #[tabled(skip)]
    pub batches: Vec<BatchStats>,
}

#[derive(Tabled)]
pub struct BatchStats {
    pub batch_idx: usize,
    pub num_timesteps: usize,
    pub num_sequences: usize,
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        self.get_item(index)
    }

    fn len(&self) -> usize {
        self.sequences
    }
}
