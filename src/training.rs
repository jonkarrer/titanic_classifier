use burn::{
    config::Config,
    module::AutodiffModule,
    nn::loss::MseLoss,
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::Backend,
    tensor::{activation::sigmoid, backend::AutodiffBackend, ElementConversion, Int, Tensor},
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
    let test_set: DataSet<<B as AutodiffBackend>::InnerBackend> = DataSet::testing(&device);
    let mut optim = config.optimizer.init();

    for epoch in 0..20 {
        // training
        let batch = training_set.batch();
        let t = batch.targets.clone();
        let targets: Tensor<B, 2> = batch.targets.unsqueeze();
        let output: Tensor<B, 2> = model.forward(batch.inputs);
        let acc = accuracy(output.clone(), t);

        let loss = MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        println!(
            "[Train - Epoch {}] Loss {:.3} | Accuracy {:.3} %",
            epoch,
            loss.clone().into_scalar(),
            acc,
        );

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(5e-5, model, grads);

        // validation
        let model_valid = model.valid();
        let batch = test_set.batch();
        let t = batch.targets.clone();
        let targets = batch.targets.unsqueeze();
        let inputs = batch.inputs;
        let output = model_valid.forward(inputs);

        let loss = MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        let acc = accuracy(output, t);

        println!(
            "*** [Validate - Epoch {}] Loss {:.3} | Accuracy {:.3} %",
            epoch,
            loss.clone().into_scalar(),
            acc,
        );
    }
}

fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1>) -> f32 {
    let output = sigmoid(output);
    let predictions: Tensor<B, 1, Int> = output.greater_elem(0.5).int().squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets.int()).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}
