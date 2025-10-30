use burn::data::dataset::Dataset;

pub type Timestep = [f32; 8];

pub struct SequenceDatasetItem {
    // [timesteps, sensors]
    pub sequence: Vec<Timestep>,
    pub target: Timestep,
}

pub struct SequenceDataset {
    sequence_length: usize,
    prediction_horizon: usize,
    features: Vec<Timestep>,
}

impl SequenceDataset {
    pub fn new(sequence_length: usize, prediction_horizon: usize) -> Self {
        Self {
            sequence_length,
            prediction_horizon,
            features: Vec::new(),
        }
    }

    /// Appends a single timestep to this dataset
    pub fn push(&mut self, record: Timestep) {
        self.features.push(record)
    }

    pub fn get_item(&self, index: usize) -> Option<SequenceDatasetItem> {
        // Compute the target index
        let target_idx = index + self.sequence_length + self.prediction_horizon;

        // Return None if there's not enough data for both sequence and target
        if target_idx >= self.features.len() {
            return None;
        }

        let sequence = self.features[index..index + self.sequence_length].to_vec();
        let target = self.features[target_idx].clone();

        Some(SequenceDatasetItem { sequence, target })
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        self.get_item(index)
    }

    fn len(&self) -> usize {
        if self.features.len() < self.sequence_length + self.prediction_horizon {
            0
        } else {
            self.features.len() - self.sequence_length - self.prediction_horizon
        }
    }
}
