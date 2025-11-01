use super::parser;
use crate::{
    dataset::{NUMBER_OF_SENSORS, SequenceDataset},
    parser::Analyte,
    preprocessor::{Node, Pipeline},
};

pub fn load_and_preprocess(
    paths: &[&str],
    sequence_length: usize,
    prediction_horizon: usize,
    ema_window: usize,
    ema_alpha: f32,
    zscore_window: usize,
) -> anyhow::Result<SequenceDataset> {
    let p = parser::LineParser::default();
    let mut dataset = SequenceDataset::new(sequence_length, prediction_horizon);

    for path in paths.iter() {
        dataset.new_batch();

        let mut pipelines: Vec<Pipeline> = (0..NUMBER_OF_SENSORS)
            .map(|_| {
                Pipeline::new([
                    Node::ema(ema_window, ema_alpha),
                    Node::zscore(zscore_window),
                ])
            })
            .collect();

        for row in parser::read_dat_file(path, &p)? {
            match row.gas {
                Analyte::Ethanol => {
                    // Its important not to short circuit this loop, or pipelines will not saturate at the same time.
                    let mut outputs = Vec::new();
                    for (i, dr) in row.dr.iter().enumerate() {
                        let out = pipelines[i].process(*dr);
                        outputs.push(out);
                    }
                    let preprocessed = outputs.into_iter().collect::<Option<Vec<f32>>>();

                    if let Some(values) = preprocessed {
                        if values.len() == NUMBER_OF_SENSORS {
                            let timestep: [f32; NUMBER_OF_SENSORS] = values.try_into().unwrap();
                            dataset.push(timestep);
                        }
                    }
                }
                _ => continue,
            }
        }
    }

    dataset.finalize();
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use crate::loader::load_and_preprocess;

    #[test]
    fn it_loads_and_preprocesses_a_single_batch_file() {
        let sequence_length = 3;
        let prediction_horizon = 1;
        let ema_window_size = 3;
        let ema_alpha = 0.5;
        let zscore_window_size = 3;

        let dataset = load_and_preprocess(
            &["data/test_batch_1.dat"],
            sequence_length,
            prediction_horizon,
            ema_window_size,
            ema_alpha,
            zscore_window_size,
        )
        .expect("dataset should load");

        let stats = dataset.stats();
        assert_eq!(stats.num_batches, 1);
        assert_eq!(stats.total_sequences, 3);
        assert_eq!(stats.sequence_length, 3);
        assert_eq!(stats.prediction_horizon, 1);
        assert_eq!(stats.batches.len(), 1);

        assert_eq!(stats.batches[0].batch_idx, 1);
        assert_eq!(stats.batches[0].num_timesteps, 6);
        assert_eq!(stats.batches[0].num_sequences, 3);
    }

    #[test]
    fn it_loads_and_preprocesses_mutliple_batch_files() {
        let sequence_length = 3;
        let prediction_horizon = 1;
        let ema_window_size = 3;
        let ema_alpha = 0.5;
        let zscore_window_size = 3;

        let dataset = load_and_preprocess(
            &["data/test_batch_1.dat", "data/test_batch_2.dat"],
            sequence_length,
            prediction_horizon,
            ema_window_size,
            ema_alpha,
            zscore_window_size,
        )
        .expect("dataset should load");

        let stats = dataset.stats();
        assert_eq!(stats.num_batches, 2);
        assert_eq!(stats.total_sequences, 5);
        assert_eq!(stats.sequence_length, 3);
        assert_eq!(stats.prediction_horizon, 1);
        assert_eq!(stats.batches.len(), 2);

        assert_eq!(stats.batches[0].batch_idx, 1);
        assert_eq!(stats.batches[0].num_timesteps, 6);
        assert_eq!(stats.batches[0].num_sequences, 3);

        assert_eq!(stats.batches[1].batch_idx, 2);
        assert_eq!(stats.batches[1].num_timesteps, 5);
        assert_eq!(stats.batches[1].num_sequences, 2);
    }
}
