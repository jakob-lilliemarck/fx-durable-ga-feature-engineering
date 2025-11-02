use crate::preprocessor::Pipeline;
use burn::data::dataset::Dataset;
use std::collections::HashMap;

pub type Timestep = Vec<f32>;

pub struct SequenceDatasetItem {
    pub sequence: Vec<Timestep>,
    pub target: Timestep,
}

pub struct SequenceDataset {
    sequence_length: usize,
    prediction_horizon: usize,
    features: Vec<Timestep>,
}

pub struct DatasetBuilder<'a> {
    pipelines: HashMap<&'a str, Pipeline>,
    features: &'a [&'a str],
    cache: HashMap<String, f32>,
    dataset: Vec<Vec<f32>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No pipeline configured for feature \"{0}\"")]
    MissingPipeline(String),
    #[error("Record has no value at key \"{0}\"")]
    MissingValue(String),
    #[error("Split factor must be a number between 0.0 and 1.0, got: {0}")]
    InvalidSplitFactor(f32),
}

impl<'a> DatasetBuilder<'a> {
    pub fn new(
        pipelines: HashMap<&'a str, Pipeline>,
        features: &'a [&'a str],
        size_hint: Option<usize>,
    ) -> Self {
        Self {
            pipelines,
            features,
            cache: HashMap::new(),
            dataset: Vec::with_capacity(size_hint.unwrap_or(0)),
        }
    }

    // NOTE!
    // - Required records to be pushed IN ORDER!
    // - Currently does not forward fill
    pub fn push(&mut self, record: HashMap<String, f32>) -> Result<(), Error> {
        // Create a timestep vector
        let mut timestep = Vec::with_capacity(self.features.len());

        for feature in self.features.iter() {
            // Try to get the pipeline, error out on not found
            let pipeline = match self.pipelines.get_mut(feature) {
                Some(pipeline) => pipeline,
                None => return Err(Error::MissingPipeline(feature.to_string())),
            };

            // Try to get the value from the record, error out on not found
            match record.get(*feature) {
                Some(value) => {
                    // If we got something, compute a the next value and push to timestep
                    if let Some(value) = pipeline.process(value.to_owned()) {
                        self.cache.insert(feature.to_string(), value);
                        timestep.push(value)
                    }
                }
                None => {
                    // if we did not get anything, pick the last value from the the forward fill cache, if one exists
                    if let Some(value) = self.cache.get(*feature) {
                        timestep.push(value.to_owned())
                    } else {
                        // A value could not be provided through computation or forward fill - error out
                        return Err(Error::MissingValue(feature.to_string()));
                    }
                }
            };
        }

        // If a value was produced for _all_ features, push the timestep to features
        if self.features.len() == timestep.len() {
            self.dataset.push(timestep)
        };

        Ok(())
    }

    pub fn build(
        self,
        sequence_length: usize,
        prediction_horizon: usize,
        split: f32,
    ) -> Result<(SequenceDataset, SequenceDataset), Error> {
        if split > 1.0 || split < 0.0 {
            return Err(Error::InvalidSplitFactor(split));
        }

        let split_idx = (self.dataset.len() as f32 * split) as usize;

        let (training_data, validation_data) = self.dataset.split_at(split_idx);

        let training = SequenceDataset {
            sequence_length,
            prediction_horizon,
            features: training_data.to_vec(),
        };

        let validation = SequenceDataset {
            sequence_length,
            prediction_horizon,
            features: validation_data.to_vec(),
        };

        Ok((training, validation))
    }
}

impl SequenceDataset {
    /// Create a sequence item from a starting index
    pub fn create_sequence_item(&self, index: usize) -> Option<SequenceDatasetItem> {
        // Calculate where the target will be positioned
        let target_index = index + self.sequence_length + self.prediction_horizon;

        // Check if we have enough data for both sequence and target
        if target_index >= self.features.len() {
            return None;
        }

        // Get a vector of owned values for the sequence
        let sequence = self.features[index..index + self.sequence_length].to_vec();
        let target = self.features[target_index].clone();

        // Create the item without validation since data was validated on insertion
        Some(SequenceDatasetItem { sequence, target })
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        self.create_sequence_item(index)
    }

    fn len(&self) -> usize {
        self.features.len()
    }
}
