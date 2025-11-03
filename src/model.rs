use burn::nn;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct SimpleRnn<B: Backend> {
    w_ih: nn::Linear<B>,       // input → hidden
    w_hh: nn::Linear<B>,       // hidden → hidden
    linear_out: nn::Linear<B>, // hidden → output
}

impl<B: Backend> SimpleRnn<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            w_ih: nn::LinearConfig::new(input_size, hidden_size).init(device),
            w_hh: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            linear_out: nn::LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _, _] = input.dims();
        let hidden_size = self.w_hh.weight.dims()[0];

        let mut h = Tensor::<B, 2>::zeros([batch_size, hidden_size], &input.device());

        // Split along dimension 1 (sequence dimension)
        let timesteps: Vec<_> = input.split(1, 1);

        for x_t in timesteps {
            // x_t: [batch, 1, input_size] → squeeze dim 1
            let x_t = x_t.squeeze_dim(1);
            let x_proj = self.w_ih.forward(x_t);
            let h_proj = self.w_hh.forward(h);
            h = (x_proj + h_proj).tanh();
        }

        self.linear_out.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct SimpleLstm<B: Backend> {
    lstm: nn::Lstm<B>,
    linear_out: nn::Linear<B>,
}

impl<B: Backend> SimpleLstm<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            lstm: nn::LstmConfig::new(input_size, hidden_size, true).init(device),
            linear_out: nn::LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Forward pass through LSTM
        let (output, _state) = self.lstm.forward(input, None);
        let seq_len = output.dims()[1];

        let last_index = Tensor::<B, 1, Int>::from_data([seq_len as i32 - 1], &output.device());

        // Select last time step and remove singleton dimension
        let last_step = output.select(1, last_index).squeeze_dim(1); // only remove seq dim

        // Linear projection to output size
        self.linear_out.forward(last_step)
    }
}
