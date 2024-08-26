use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};

use super::RawData;

pub struct DataPoint<B: Backend> {
    label: Tensor<B, 1>,   // [0.0] or [1.0]
    feature: Tensor<B, 1>, // [f32, f32, ...]
}

pub struct DataSet<B: Backend> {
    data: Vec<DataPoint<B>>,
}

impl<B: Backend> DataSet<B> {
    pub fn new(raw_data: RawData, device: &B::Device) -> DataSet<B> {
        let mut data = Vec::new();
        for record in raw_data.get_all_rows() {
            let survived = record.survived as f32;
            let age = record.age as f32;
            let class = record.pclass as f32;
            let fare = record.fare;
            let sex: f32 = if record.sex == "male" { 1.0 } else { 0.0 };
            let parch = record.parch as f32;
            let sibsp = record.sib_sp as f32;

            let label: Tensor<B, 1> = Tensor::from_floats([survived], device);

            let feature: Tensor<B, 1> =
                Tensor::from_floats([age, class, fare, sex, parch, sibsp], device);

            let data_point = DataPoint { label, feature };

            data.push(data_point);
        }

        DataSet { data }
    }
}
