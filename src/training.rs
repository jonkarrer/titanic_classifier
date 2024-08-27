use burn::{
    config::Config,
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, ElementConversion, Int, Tensor},
};

use crate::{
    data::DataSet,
    model::{Model, ModelConfig},
};

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 10)]
    pub epochs: usize,

    #[config(default = 2)]
    pub workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: SgdConfig,

    #[config(default = 6)]
    pub input_feature_len: usize,

    #[config(default = 889)]
    pub dataset_size: usize,
}

pub fn train<B: AutodiffBackend>(device: B::Device) {
    let optimizer = SgdConfig::new();
    let config = ExpConfig::new(optimizer);
    let mut model: Model<B> = ModelConfig::new().init(&device);
    B::seed(config.seed);

    let training_set: DataSet<B> = DataSet::training(&device);
    let test_set: DataSet<B> = DataSet::testing(&device);
    let mut optim = config.optimizer.init();

    for epoch in 0..config.epochs {
        // training
        let batch = training_set.batch();
        let loss = model.forward_step(batch.inputs, batch.targets);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(1e-4, model, grads);

        println!("Loss: {}", loss.clone().into_scalar());

        // validation
        // let model_valid = model.valid();
        // let batch = test_set.batch();
        // let loss = model_valid.forward(batch.inputs);
    }
}

fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}
