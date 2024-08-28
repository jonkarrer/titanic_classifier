#![allow(dead_code)]
use burn::{
    config::Config,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::AutodiffBackend,
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

    pub optimizer: AdamConfig,

    #[config(default = 6)]
    pub input_feature_len: usize,

    #[config(default = 889)]
    pub dataset_size: usize,
}

pub fn train<B: AutodiffBackend>(model_path: &str, device: B::Device) {
    let optimizer = AdamConfig::new();
    let config = ExpConfig::new(optimizer);
    let mut model: Model<B> = ModelConfig::new().init(&device);
    B::seed(config.seed);

    let training_set: DataSet<B> = DataSet::training(&device);
    let test_set: DataSet<<B as AutodiffBackend>::InnerBackend> = DataSet::testing(&device);
    let mut optim = config.optimizer.init();

    for epoch in 0..10 {
        // training
        let batch = training_set.batch();
        let output = model.forward_step(&batch, &device);

        println!(
            "[Train - Epoch {}] Loss {:.3} | Accuracy {:.3} %",
            epoch,
            output.loss.clone().into_scalar(),
            output.accuracy
        );

        let grads = output.loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(8e-3, model, grads);

        // validation
        let model_valid = model.valid();
        let batch = test_set.batch();
        let output = model_valid.forward_step(&batch, &device);

        println!(
            "*** [Validate - Epoch {}] Loss {:.3} | Accuracy {:.3} %",
            epoch,
            output.loss.into_scalar(),
            output.accuracy,
        );
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(model_path, &recorder)
        .expect("could not save model");
}
