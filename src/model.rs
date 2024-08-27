use burn::{
    config::Config,
    module::Module,
    nn::{loss::BinaryCrossEntropyLossConfig, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor},
};

use crate::data::Batch;

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

pub struct ClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub accuracy: f32,
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

    pub fn forward_step(&self, batch: &Batch<B>, device: &B::Device) -> ClassificationOutput<B> {
        let predictions = self.forward(batch.inputs.clone());
        let labels = batch.labels.clone().int();

        let accuracy = Self::accuracy(predictions.clone(), labels.clone());

        let loss_func = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(device);
        let loss = loss_func.forward(predictions.clone(), labels.clone());

        ClassificationOutput { loss, accuracy }
    }

    fn accuracy(output: Tensor<B, 2>, targets: Tensor<B, 2, Int>) -> f32 {
        let predictions: Tensor<B, 1, Int> = output.greater_elem(0.5).int().squeeze(1);
        let num_predictions: usize = targets.dims().iter().product();
        let num_corrects = predictions
            .equal(targets.squeeze(1))
            .int()
            .sum()
            .into_scalar();

        num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
    }
}
