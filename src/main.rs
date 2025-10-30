mod batcher;
mod dataset;
mod parse;
mod preprocessing;

pub struct Preprocessor {
    nodes: [preprocessing::Node; 2],
}

impl Preprocessor {
    fn process(&mut self, value: f32) -> Option<f32> {
        let mut value: Option<f32> = Some(value);

        for node in self.nodes.iter_mut() {
            value = value.and_then(|value| node.process(value));
        }

        value
    }
}

fn main() -> anyhow::Result<()> {
    let parser = parse::LineParser::default();

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

    let mut rows: Vec<parse::Row> = Vec::new();

    for path in paths {
        for row in parse::read_dat_file(path, &parser)? {
            rows.push(row);
        }
    }

    println!("row count {:?}", rows.len());

    Ok(())
}
