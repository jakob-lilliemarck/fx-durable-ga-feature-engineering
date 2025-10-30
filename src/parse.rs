use regex::Regex;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub enum Analyte {
    Ethanol,
    Ethylene,
    Ammonia,
    Acetaldehyde,
    Acetone,
    Toluene,
}

#[derive(Debug, thiserror::Error)]
#[error("Unkown analyte: '{0}'")]
pub struct UnkownAnalyte(u8);

impl TryFrom<u8> for Analyte {
    type Error = UnkownAnalyte;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Analyte::Ethanol),
            2 => Ok(Analyte::Ethylene),
            3 => Ok(Analyte::Ammonia),
            4 => Ok(Analyte::Acetaldehyde),
            5 => Ok(Analyte::Acetone),
            6 => Ok(Analyte::Toluene),
            number => Err(UnkownAnalyte(number)),
        }
    }
}

#[derive(Debug)]
pub struct Row {
    idx: usize,
    gas: Analyte,
    dr: Vec<f64>,
    dr_abs: Vec<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("ParseError")]
    ParseError(String),
    #[error("UnknownAnalyte")]
    UnknownAnalyte(#[from] UnkownAnalyte),
}

impl TryFrom<String> for Row {
    type Error = ParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        todo!()
    }
}

pub struct LineParser {
    re_heading: Regex,
    re_feature: Regex,
}

impl Default for LineParser {
    fn default() -> Self {
        Self::new().expect("LineParser regex compilation should not fail")
    }
}

impl LineParser {
    pub fn new() -> Result<Self, regex::Error> {
        // Pattern to parse the first token only
        let re_heading = Regex::new(r"^(\d+);([\d.]+)")?;

        // Pattern to parse the feature values
        let re_feature = Regex::new(r"(\d+):([-\d.]+)")?;

        Ok(Self {
            re_heading,
            re_feature,
        })
    }

    fn parse_line(&self, line: &str, idx: usize) -> Result<Row, ParseError> {
        // Parse first token (id;value)
        let first_cap = self
            .re_heading
            .captures(line)
            .ok_or(ParseError::ParseError(
                "Failed to parse the first token".to_string(),
            ))?;

        // Parse the analyte
        let gas: Analyte = first_cap[1]
            .parse::<u8>()
            .map_err(|_| ParseError::ParseError("Failed to parse id".to_string()))?
            .try_into()?;

        let mut indexed_dr: Vec<(u32, f64)> = Vec::with_capacity(16);
        let mut indexed_dr_abs: Vec<(u32, f64)> = Vec::with_capacity(16);
        for cap in self.re_feature.captures_iter(line) {
            let feature_idx = cap[1]
                .parse::<u32>()
                .map_err(|_| ParseError::ParseError("Failed to parse feature index".to_string()))?;

            let feature_value = cap[2]
                .parse::<f64>()
                .map_err(|_| ParseError::ParseError("Failed to parse feature value".to_string()))?;

            if feature_idx % 8 == 1 {
                indexed_dr.push((feature_idx, feature_value))
            } else if feature_idx % 8 == 2 {
                indexed_dr_abs.push((feature_idx, feature_value))
            } else {
                // skipping over any other features
                continue;
            }
        }

        // Sort by index to guarantee ordering
        indexed_dr.sort_unstable_by_key(|&(idx, _)| idx);
        indexed_dr_abs.sort_unstable_by_key(|&(idx, _)| idx);

        let dr: Vec<f64> = indexed_dr.into_iter().map(|(_, value)| value).collect();
        let dr_abs: Vec<f64> = indexed_dr_abs.into_iter().map(|(_, value)| value).collect();

        Ok(Row {
            idx,
            gas,
            dr,
            dr_abs,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO Error")]
    IOError(#[from] io::Error),
    #[error("Failed to parse file '{0}'")]
    ParseError(#[from] ParseError),
}

pub fn read_dat_file<P: AsRef<Path>>(path: P, parser: &LineParser) -> Result<Vec<Row>, Error> {
    let file = File::open(path)?;

    let reader = BufReader::new(file);

    let mut rows = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        // Error out on failure
        let line = line?;

        // Skip over empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Convert to row type
        let row = parser.parse_line(&line, idx)?;

        // Push to buffer
        rows.push(row);
    }

    Ok(rows)
}
