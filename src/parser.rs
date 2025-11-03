use csv::{Reader, StringRecord};
use std::fs::File;
use std::io;
use std::num::{ParseFloatError, ParseIntError};

#[derive(Debug)]
pub enum WindDirection {
    N,
    NNE,
    NE,
    ENE,
    E,
    ESE,
    SE,
    SSE,
    S,
    SSW,
    SW,
    WSW,
    W,
    WNW,
    NW,
    NNW,
}

#[derive(Debug, thiserror::Error)]
#[error("Unknown wind direction: {0}")]
pub struct UnknownWindDirection(String);

impl TryFrom<&str> for WindDirection {
    type Error = UnknownWindDirection;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "N" => Ok(WindDirection::N),
            "NNE" => Ok(WindDirection::NNE),
            "NE" => Ok(WindDirection::NE),
            "ENE" => Ok(WindDirection::ENE),
            "E" => Ok(WindDirection::E),
            "ESE" => Ok(WindDirection::ESE),
            "SE" => Ok(WindDirection::SE),
            "SSE" => Ok(WindDirection::SSE),
            "S" => Ok(WindDirection::S),
            "SSW" => Ok(WindDirection::SSW),
            "SW" => Ok(WindDirection::SW),
            "WSW" => Ok(WindDirection::WSW),
            "W" => Ok(WindDirection::W),
            "WNW" => Ok(WindDirection::WNW),
            "NW" => Ok(WindDirection::NW),
            "NNW" => Ok(WindDirection::NNW),
            _ => Err(UnknownWindDirection(value.to_string())),
        }
    }
}

#[derive(Debug)]
pub struct Row {
    pub no: u32,
    pub day: f32,
    pub hour: f32,
    pub station: String,
    pub pm2_5: Option<f32>,
    pub pm10: Option<f32>,
    pub so2: Option<f32>,
    pub no2: Option<f32>,
    pub co: Option<f32>,
    pub o3: Option<f32>,
    pub temp: Option<f32>,
    pub pres: Option<f32>,
    pub dewp: Option<f32>,
    pub rain: Option<f32>,
    pub wd: Option<WindDirection>,
    pub wspm: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO Error")]
    IOError(#[from] io::Error),
    #[error("Parsing error")]
    ParseError(Box<dyn std::error::Error + Send + Sync>),
    #[error("Unknown wind direction: {0}")]
    UnknownWindDirection(#[from] UnknownWindDirection),
}

impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Self {
        Error::ParseError(Box::new(err))
    }
}

impl From<ParseFloatError> for Error {
    fn from(err: ParseFloatError) -> Self {
        Error::ParseError(Box::new(err))
    }
}

impl From<ParseIntError> for Error {
    fn from(err: ParseIntError) -> Self {
        Error::ParseError(Box::new(err))
    }
}

pub struct RowIterator {
    reader: Reader<File>,
}

impl RowIterator {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path)?;

        let reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        Ok(RowIterator { reader })
    }
}

impl Iterator for RowIterator {
    type Item = Result<Row, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.reader.records().next()? {
            Ok(r) => r,
            Err(err) => return Some(Err(Error::ParseError(Box::new(err)))),
        };

        Some(parse_row(&record))
    }
}

fn parse_optional_f32(s: &str) -> Result<Option<f32>, Error> {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
        Ok(None)
    } else {
        let value = trimmed.parse::<f32>()?;
        Ok(Some(value))
    }
}

fn parse_wind_direction(s: &str) -> Result<Option<WindDirection>, Error> {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "NA" {
        Ok(None)
    } else {
        Ok(Some(WindDirection::try_from(trimmed)?))
    }
}

fn parse_row(record: &StringRecord) -> Result<Row, Error> {
    // Parse day and hour as f32 directly
    let day: f32 = record[3].parse()?;
    let hour: f32 = record[4].parse()?;

    Ok(Row {
        no: record[0].parse()?,
        day,
        hour,
        station: record[17].to_string(),
        pm2_5: parse_optional_f32(&record[5])?,
        pm10: parse_optional_f32(&record[6])?,
        so2: parse_optional_f32(&record[7])?,
        no2: parse_optional_f32(&record[8])?,
        co: parse_optional_f32(&record[9])?,
        o3: parse_optional_f32(&record[10])?,
        temp: parse_optional_f32(&record[11])?,
        pres: parse_optional_f32(&record[12])?,
        dewp: parse_optional_f32(&record[13])?,
        rain: parse_optional_f32(&record[14])?,
        wd: parse_wind_direction(&record[15])?,
        wspm: parse_optional_f32(&record[16])?,
    })
}

// Get an iterator over the rows of the CSV file
pub fn read_csv(path: &str) -> Result<impl Iterator<Item = Result<Row, Error>>, Error> {
    Ok(RowIterator::new(path)?)
}
