use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};

use super::RawData;

pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 2>,  // [[f32, f32, ...], [f32, f32, ...], ...]
    pub targets: Tensor<B, 1>, // [1.0,0.0,0.0,1.0,...]
}

pub struct DataPoint<B: Backend> {
    pub label: Tensor<B, 1>,   // [0.0] or [1.0]
    pub feature: Tensor<B, 1>, // [f32, f32, ...]
}

impl<B: Backend> DataPoint<B> {
    pub fn new(label: [f32; 1], feature: [f32; 6], device: &B::Device) -> DataPoint<B> {
        let label: Tensor<B, 1> = Tensor::from_floats(label, device);

        let feature: Tensor<B, 1> = Tensor::from_floats(feature, device);
        DataPoint { label, feature }
    }
}

pub struct DataSet<B: Backend> {
    pub data: Vec<DataPoint<B>>,
}

impl<B: Backend> DataSet<B> {
    pub fn training(device: &B::Device) -> DataSet<B> {
        let raw_data = RawData::new("data/train.csv");
        Self::new(raw_data, device)
    }

    pub fn testing(device: &B::Device) -> DataSet<B> {
        let raw_data = RawData::new("data/validation.csv");
        Self::new(raw_data, device)
    }

    fn new(raw_data: RawData, device: &B::Device) -> DataSet<B> {
        let mut data = Vec::new();
        for record in raw_data.get_all_rows() {
            let survived = record.survived as f32;
            let age = record.age as f32;
            let class = record.pclass as f32;
            let fare = record.fare;
            let sex: f32 = if record.sex == "male" { 1.0 } else { 0.0 };
            let parch = record.parch as f32;
            let sibsp = record.sib_sp as f32;

            let label = [survived];
            let feature = [age, class, fare, sex, parch, sibsp];

            data.push(DataPoint::new(label, feature, device));
        }

        DataSet { data }
    }

    pub fn batch(&self) -> Batch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut labels: Vec<Tensor<B, 1>> = Vec::new();

        for dp in &self.data {
            inputs.push(dp.feature.clone().unsqueeze());
            labels.push(dp.label.clone());
        }

        let inputs = Tensor::cat(inputs, 0);
        let labels = Tensor::cat(labels, 0);
        Batch {
            inputs,
            targets: labels,
        }
    }
}
