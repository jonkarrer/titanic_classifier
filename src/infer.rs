#![allow(dead_code)]
use burn::{
    module::Module,
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};

use crate::{
    data,
    model::{ModelConfig, ModelRecord},
};

pub fn infer<B: Backend>(model_path: &str, device: &B::Device) {
    let record: ModelRecord<B> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), device)
        .expect("could not load model");
    let model = ModelConfig::new().init(device).load_record(record);

    let data = data::TestDataSet::<B>::new(device);
    let batch = data.batch();
    let output = model.interface(&batch);

    let file = std::fs::File::create("data/submission.csv").expect("could not create file");

    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&["PassengerId", "Survived"])
        .expect("could not write header");

    for item in output {
        wtr.write_record(&[&item.0.to_string(), &item.1.to_string()])
            .expect("could not write record");
    }
}
