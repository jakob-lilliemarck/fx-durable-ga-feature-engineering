use std::collections::VecDeque;

pub struct Ema {
    /// The window size of this ema
    window: usize,
    /// Ema alpha, the "smoothing" factor
    alpha: f32,
    /// The current value
    value: f32,
    /// Count to track window size saturation
    count: usize,
}

impl Ema {
    fn new(window: usize, alpha: f32) -> Self {
        Self {
            window,
            alpha,
            value: 0.0,
            count: 0,
        }
    }

    fn process(&mut self, value: f32) -> Option<f32> {
        if self.count == 0 {
            // Initialize with first value
            self.value = value;
        } else {
            // Apply EMA formula: EMA = value * multiplier + previous_ema * (1 - multiplier)
            self.value = value * self.alpha + self.value * (1.0 - self.alpha);
        }

        self.count += 1;

        // Return None until window is saturated
        if self.count < self.window {
            None
        } else {
            Some(self.value)
        }
    }
}

pub struct ZScore {
    window: usize,
    values: VecDeque<f32>, // Ring buffer of recent values
    count: usize,
}

impl ZScore {
    fn new(window: usize) -> Self {
        Self {
            window,
            values: VecDeque::with_capacity(window),
            count: 0,
        }
    }

    fn process(&mut self, value: f32) -> Option<f32> {
        self.values.push_back(value);
        if self.values.len() > self.window {
            self.values.pop_front();
        }
        self.count += 1;

        // If the window is not saturated, return None
        if self.count < self.window {
            return None;
        }

        // Calculate the SMA
        let mean = self.values.iter().sum::<f32>() / self.window as f32;

        // Calculate variance
        let variance =
            self.values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / self.values.len() as f32;

        // Calculate the standard deviation
        let std = variance.sqrt();

        Some((value - mean) / std)
    }
}

pub enum Node {
    Noop,
    Ema(Ema),
    ZScore(ZScore),
}

impl Node {
    pub fn ema(window: usize, alpha: f32) -> Self {
        Self::Ema(Ema::new(window, alpha))
    }

    pub fn zscore(window: usize) -> Self {
        Self::ZScore(ZScore::new(window))
    }

    pub fn noop() -> Self {
        Self::Noop
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        match self {
            Self::Noop => Some(value),
            Self::Ema(ema) => ema.process(value),
            Self::ZScore(zscore) => zscore.process(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn it_computes_ema() {
        let mut ema = Ema::new(3, 0.5);

        assert_eq!(ema.process(10.0), None); // value=10.0
        assert_eq!(ema.process(20.0), None); // value=15.0
        assert_relative_eq!(ema.process(30.0).unwrap(), 22.5); // value=22.5
        assert_relative_eq!(ema.process(40.0).unwrap(), 31.25); // value=31.25
    }

    #[test]
    fn it_computes_zscore() {
        let mut zscore = ZScore::new(3);

        assert_eq!(zscore.process(10.0), None);
        assert_eq!(zscore.process(20.0), None);

        // Window [10, 20, 30]: mean=20, std≈8.165, z=(30-20)/8.165≈1.225
        assert_relative_eq!(zscore.process(30.0).unwrap(), 1.225, epsilon = 0.01);

        // Window [20, 30, 40]: mean=30, std≈8.165, z=(40-30)/8.165≈1.225
        assert_relative_eq!(zscore.process(40.0).unwrap(), 1.225, epsilon = 0.01);
    }

    #[test]
    fn it_computes_zscore_of_ema() {
        let mut ema = Ema::new(2, 0.5);
        let mut zscore = ZScore::new(2);

        assert_eq!(ema.process(10.0), None); // ema: 10.0
        let ema1 = ema.process(20.0).unwrap(); // ema: 15.0
        let ema2 = ema.process(30.0).unwrap(); // ema: 22.5
        let ema3 = ema.process(40.0).unwrap(); // ema: 31.25

        assert_eq!(zscore.process(ema1), None); // warmup

        // Window [15.0, 22.5]: mean=18.75, std=3.75, z=(22.5-18.75)/3.75=1.0
        let z1 = zscore.process(ema2).unwrap();
        assert_relative_eq!(z1, 1.0, epsilon = 0.01);

        // Window [22.5, 31.25]: mean=26.875, std=4.375, z=(31.25-26.875)/4.375=1.0
        let z2 = zscore.process(ema3).unwrap();
        assert_relative_eq!(z2, 1.0, epsilon = 0.01);
    }
}
