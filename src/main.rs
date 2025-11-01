use clap::Parser;

mod batcher;
mod dataset;
mod loader;
mod model;
mod parser;
mod preprocessor;
mod train;

/// Train a RNN on "Gas sensor array draft dataset"
#[derive(Debug, Parser)]
#[command(
    name = "example-rnn-gas-sensor-array",
    version = "1.0",
    author = "Jakob",
    about = "Give credits"
)]
struct Cli {
    // ============================================================
    // Model params
    // ============================================================
    /// Hidden layer size
    #[arg(long, required = true)]
    input_size: usize,

    /// Hidden layer size
    #[arg(long, required = true)]
    hidden_size: usize,

    /// Hidden layer size
    #[arg(long, required = true)]
    output_size: usize,

    /// Learning rate
    #[arg(long, required = true)]
    learning_rate: f64,

    // ============================================================
    // Preprocessing params
    // ============================================================
    /// Ema window size
    #[clap(long, required = true)]
    ema_window: usize,

    /// Ema alpha
    #[clap(long, required = true)]
    ema_alpha: f32,

    /// Zscore window size
    #[clap(long, required = true)]
    zscore_window: usize,

    // ============================================================
    // Training params
    // ============================================================
    /// Epochs
    #[clap(long, default_value_t = 100)]
    epochs: usize,
}

fn main() -> anyhow::Result<()> {
    let parser = parser::LineParser::default();

    let paths = [
        "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch1.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch2.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch3.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch4.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch5.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch6.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch7.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch8.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch9.dat",
        // "data/gas+sensor+array+drift+dataset+at+different+concentrations/batch10.dat",
    ];

    let mut rows: Vec<parser::Row> = Vec::new();

    for path in paths {
        for row in parser::read_dat_file(path, &parser)? {
            rows.push(row);
        }
    }

    println!("row count {:?}", rows.len());

    Ok(())
}
