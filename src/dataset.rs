use crate::preprocessor::Pipeline;
use burn::data::dataset::Dataset;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub type Timestep = Vec<f32>;

pub struct DatasetBuilder {
    pipelines: HashMap<String, Pipeline>,
    output_names: Vec<String>,
    source_columns: Vec<String>,
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
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl DatasetBuilder {
    pub fn new(
        pipelines: HashMap<String, Pipeline>,
        output_names: Vec<String>,
        source_columns: Vec<String>,
        size_hint: Option<usize>,
    ) -> Self {
        Self {
            pipelines,
            output_names,
            source_columns,
            cache: HashMap::new(),
            dataset: Vec::with_capacity(size_hint.unwrap_or(0)),
        }
    }

    // NOTE!
    // - Required records to be pushed IN ORDER!
    // - Currently does not forward fill
    pub fn push(&mut self, record: HashMap<String, f32>) -> Result<(), Error> {
        // Create a timestep vector
        let mut timestep = Vec::with_capacity(self.output_names.len());

        for (output_name, source_column) in self.output_names.iter().zip(self.source_columns.iter())
        {
            // Try to get the pipeline by OUTPUT name, error out on not found
            let pipeline = match self.pipelines.get_mut(output_name) {
                Some(pipeline) => pipeline,
                None => return Err(Error::MissingPipeline(output_name.to_string())),
            };

            // Try to get the value from the record using SOURCE column
            match record.get(source_column) {
                Some(value) => {
                    // If we got something, compute the next value and push to timestep
                    if let Some(value) = pipeline.process(value.to_owned()) {
                        self.cache.insert(output_name.to_string(), value);
                        timestep.push(value)
                    }
                }
                None => {
                    // if we did not get anything, pick the last value from the forward fill cache, if one exists
                    if let Some(value) = self.cache.get(output_name) {
                        timestep.push(value.to_owned())
                    } else {
                        // A value could not be provided through computation or forward fill - error out
                        return Err(Error::MissingValue(source_column.to_string()));
                    }
                }
            };
        }

        // If a value was produced for _all_ features, push the timestep to features
        if self.output_names.len() == timestep.len() {
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

        // Pre-materialize all items during dataset creation
        let training = SequenceDataset::from_features(
            training_data.to_vec(),
            sequence_length,
            prediction_horizon,
        );

        let validation = SequenceDataset::from_features(
            validation_data.to_vec(),
            sequence_length,
            prediction_horizon,
        );

        Ok((training, validation))
    }

    /// Dump the dataset to a CSV file
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut file = File::create(path)?;

        // Write header (semicolon-separated for European locale)
        let header = self.output_names.join(";");
        writeln!(file, "{}", header)?;

        // Write each timestep as a row (with comma as decimal separator)
        for timestep in &self.dataset {
            let row: Vec<String> = timestep
                .iter()
                .map(|v| v.to_string().replace('.', ","))
                .collect();
            writeln!(file, "{}", row.join(";"))?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct SequenceDatasetItem {
    pub sequence: Vec<Timestep>,
    pub target: Timestep,
}

pub struct SequenceDataset {
    items: Vec<SequenceDatasetItem>,
}

impl SequenceDataset {
    /// Build dataset from features by pre-creating all items
    pub fn from_features(
        features: Vec<Timestep>,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Self {
        let mut items = Vec::new();

        // Pre-create all sequence items once
        for index in 0..features.len() {
            let target_index = index + sequence_length + prediction_horizon;

            if target_index >= features.len() {
                break;
            }

            let sequence = features[index..index + sequence_length].to_vec();
            let target = features[target_index].clone();

            items.push(SequenceDatasetItem { sequence, target });
        }

        Self { items }
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        // Just clone the pre-created item - much faster than creating from scratch
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
