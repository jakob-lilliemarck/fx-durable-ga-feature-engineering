use crate::preprocessor::Pipeline;
use burn::data::dataset::Dataset;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub type Timestep = Vec<f32>;

pub struct DatasetBuilder {
    feature_pipelines: HashMap<String, Pipeline>,
    feature_output_names: Vec<String>,
    feature_source_columns: Vec<String>,
    feature_cache: HashMap<String, f32>,

    target_pipelines: HashMap<String, Pipeline>,
    target_output_names: Vec<String>,
    target_source_columns: Vec<String>,
    target_cache: HashMap<String, f32>,

    features: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
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
        feature_pipelines: HashMap<String, Pipeline>,
        feature_output_names: Vec<String>,
        feature_source_columns: Vec<String>,
        target_pipelines: HashMap<String, Pipeline>,
        target_output_names: Vec<String>,
        target_source_columns: Vec<String>,
        size_hint: Option<usize>,
    ) -> Self {
        Self {
            feature_pipelines,
            feature_output_names,
            feature_source_columns,
            feature_cache: HashMap::new(),

            target_pipelines,
            target_output_names,
            target_source_columns,
            target_cache: HashMap::new(),

            features: Vec::with_capacity(size_hint.unwrap_or(0)),
            targets: Vec::with_capacity(size_hint.unwrap_or(0)),
        }
    }

    // NOTE!
    // - Required records to be pushed IN ORDER!
    // - Currently does not forward fill
    pub fn push(&mut self, record: HashMap<String, f32>) -> Result<(), Error> {
        // Process features
        let mut feature_timestep = Vec::with_capacity(self.feature_output_names.len());

        for (output_name, source_column) in self
            .feature_output_names
            .iter()
            .zip(self.feature_source_columns.iter())
        {
            let pipeline = match self.feature_pipelines.get_mut(output_name) {
                Some(pipeline) => pipeline,
                None => return Err(Error::MissingPipeline(output_name.to_string())),
            };

            match record.get(source_column) {
                Some(value) => {
                    if let Some(value) = pipeline.process(value.to_owned()) {
                        self.feature_cache.insert(output_name.to_string(), value);
                        feature_timestep.push(value)
                    }
                }
                None => {
                    if let Some(value) = self.feature_cache.get(output_name) {
                        feature_timestep.push(value.to_owned())
                    } else {
                        return Err(Error::MissingValue(source_column.to_string()));
                    }
                }
            };
        }

        // Process targets
        let mut target_timestep = Vec::with_capacity(self.target_output_names.len());

        for (output_name, source_column) in self
            .target_output_names
            .iter()
            .zip(self.target_source_columns.iter())
        {
            let pipeline = match self.target_pipelines.get_mut(output_name) {
                Some(pipeline) => pipeline,
                None => return Err(Error::MissingPipeline(output_name.to_string())),
            };

            match record.get(source_column) {
                Some(value) => {
                    if let Some(value) = pipeline.process(value.to_owned()) {
                        self.target_cache.insert(output_name.to_string(), value);
                        target_timestep.push(value)
                    }
                }
                None => {
                    if let Some(value) = self.target_cache.get(output_name) {
                        target_timestep.push(value.to_owned())
                    } else {
                        return Err(Error::MissingValue(source_column.to_string()));
                    }
                }
            };
        }

        // Only push if we got all features AND all targets
        if self.feature_output_names.len() == feature_timestep.len()
            && self.target_output_names.len() == target_timestep.len()
        {
            self.features.push(feature_timestep);
            self.targets.push(target_timestep);
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

        let split_idx = (self.features.len() as f32 * split) as usize;

        let (training_features, validation_features) = self.features.split_at(split_idx);
        let (training_targets, validation_targets) = self.targets.split_at(split_idx);

        // Pre-materialize all items during dataset creation
        let training = SequenceDataset::from_features_and_targets(
            training_features.to_vec(),
            training_targets.to_vec(),
            sequence_length,
            prediction_horizon,
        );

        let validation = SequenceDataset::from_features_and_targets(
            validation_features.to_vec(),
            validation_targets.to_vec(),
            sequence_length,
            prediction_horizon,
        );

        Ok((training, validation))
    }

    /// Dump the dataset to a CSV file
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut file = File::create(path)?;

        // Write header (semicolon-separated for European locale)
        let mut header = self.feature_output_names.clone();
        header.extend(self.target_output_names.clone());
        writeln!(file, "{}", header.join(";"))?;

        // Write each timestep as a row (with comma as decimal separator)
        for (features, targets) in self.features.iter().zip(self.targets.iter()) {
            let mut row: Vec<String> = features
                .iter()
                .map(|v| v.to_string().replace('.', ","))
                .collect();
            row.extend(targets.iter().map(|v| v.to_string().replace('.', ",")));
            writeln!(file, "{}", row.join(";"))?;
        }

        Ok(())
    }
}

/// A single, self-contained training example.
///
/// Each item represents a complete sequence-to-target mapping and is independent
/// of all other items. This means:
/// - All timesteps in `sequence` must come from the same continuous time series
///   (e.g., same weather station, no gaps)
/// - The `target` must be the natural continuation of that sequence
/// - Items from different sources (stations, time periods) can be safely mixed
///   in the same dataset/batch
///
/// # Example
/// Valid batch composition:
/// - Item 0: Station A, hours 100-123 → predict hour 124
/// - Item 1: Station B, hours 50-73 → predict hour 74
/// - Item 2: Station A, hours 200-223 → predict hour 224
///
/// Each item is internally consistent, so mixing them doesn't create artificial
/// sequences that span different contexts.
#[derive(Clone)]
pub struct SequenceDatasetItem {
    pub sequence: Vec<Timestep>,
    pub target: Timestep,
}

pub struct SequenceDataset {
    items: Vec<SequenceDatasetItem>,
}

impl SequenceDataset {
    /// Build dataset from separate features and targets by pre-creating all items
    pub fn from_features_and_targets(
        features: Vec<Timestep>,
        targets: Vec<Timestep>,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Self {
        let mut items = Vec::new();

        // Pre-create all sequence items once
        for index in 0..features.len() {
            let target_index = index + sequence_length + prediction_horizon;

            if target_index >= features.len() || target_index >= targets.len() {
                break;
            }

            let sequence = features[index..index + sequence_length].to_vec();
            let target = targets[target_index].clone();

            items.push(SequenceDatasetItem { sequence, target });
        }

        Self { items }
    }

    /// Build dataset from pre-created items.
    ///
    /// This is useful for combining items from multiple sources (e.g., different
    /// weather stations) where each source creates its own internally-consistent
    /// sequences, but they can be safely mixed for training.
    pub fn from_items(items: Vec<SequenceDatasetItem>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::Pipeline;

    #[test]
    fn test_dataset_builder_separates_features_and_targets() {
        // Create simple pipelines (no preprocessing for clarity)
        let mut feature_pipelines = HashMap::new();
        feature_pipelines.insert("feat1".to_string(), Pipeline::new(vec![]));
        feature_pipelines.insert("feat2".to_string(), Pipeline::new(vec![]));

        let mut target_pipelines = HashMap::new();
        target_pipelines.insert("target".to_string(), Pipeline::new(vec![]));

        let mut builder = DatasetBuilder::new(
            feature_pipelines,
            vec!["feat1".to_string(), "feat2".to_string()],
            vec!["col1".to_string(), "col2".to_string()],
            target_pipelines,
            vec!["target".to_string()],
            vec!["col3".to_string()],
            None,
        );

        // Push simple records: [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...
        for i in 0..5 {
            let mut record = HashMap::new();
            record.insert("col1".to_string(), (i * 3 + 1) as f32);
            record.insert("col2".to_string(), (i * 3 + 2) as f32);
            record.insert("col3".to_string(), (i * 3 + 3) as f32);
            builder.push(record).unwrap();
        }

        // Verify features and targets are separate
        assert_eq!(builder.features.len(), 5);
        assert_eq!(builder.targets.len(), 5);

        // Check first timestep: features=[1.0, 2.0], target=[3.0]
        assert_eq!(builder.features[0], vec![1.0, 2.0]);
        assert_eq!(builder.targets[0], vec![3.0]);

        // Check second timestep: features=[4.0, 5.0], target=[6.0]
        assert_eq!(builder.features[1], vec![4.0, 5.0]);
        assert_eq!(builder.targets[1], vec![6.0]);
    }

    #[test]
    fn test_sequence_dataset_uses_separate_targets() {
        // Features: [[1, 2], [3, 4], [5, 6], [7, 8]]
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        // Targets: [[10], [20], [30], [40]]
        let targets = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0]];

        let dataset = SequenceDataset::from_features_and_targets(
            features, targets, 2, // sequence_length
            0, // prediction_horizon (predict immediately after sequence)
        );

        // Should have 2 items: [0,1]->target[2], [1,2]->target[3]
        assert_eq!(dataset.len(), 2);

        // First item: sequence=[[1,2], [3,4]], target=[30]
        let item0 = dataset.get(0).unwrap();
        assert_eq!(item0.sequence, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(item0.target, vec![30.0]);

        // Second item: sequence=[[3,4], [5,6]], target=[40]
        let item1 = dataset.get(1).unwrap();
        assert_eq!(item1.sequence, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(item1.target, vec![40.0]);
    }
}
