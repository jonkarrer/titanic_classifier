use std::process::Output;

use burn::{
    config::Config,
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor},
};

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = 6)]
    pub feature_size: usize,

    #[config(default = 64)]
    pub hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_layer = LinearConfig::new(self.feature_size, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(true)
            .init(device);
        let activation = Relu::new();

        Model {
            input_layer,
            activation,
            output_layer,
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    activation: Relu,
    output_layer: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, features: Tensor<B, 2>, labels: Tensor<B, 1>) -> Tensor<B, 1> {
        let targets: Tensor<B, 2> = labels.unsqueeze();
        let output: Tensor<B, 2> = self.forward(features);

        MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        )
    }
}
