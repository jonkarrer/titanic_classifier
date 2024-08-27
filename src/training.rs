use burn::{
    config::Config,
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer, SgdConfig},
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

    pub optimizer: SgdConfig,

    #[config(default = 6)]
    pub input_feature_len: usize,

    #[config(default = 889)]
    pub dataset_size: usize,
}

pub fn train<B: AutodiffBackend>(device: B::Device) {
    let optimizer = SgdConfig::new();
    let optimizer = optimizer
        .with_gradient_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));
    let config = ExpConfig::new(optimizer);
    let mut model: Model<B> = ModelConfig::new().init(&device);
    B::seed(config.seed);

    let training_set: DataSet<B> = DataSet::training(&device);
    let test_set: DataSet<<B as AutodiffBackend>::InnerBackend> = DataSet::testing(&device);
    let mut optim = config.optimizer.init();

    for epoch in 0..15 {
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
        model = optim.step(7e-3, model, grads);

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
}
