#![allow(dead_code)]
use burn::{prelude::Backend, tensor::Tensor};
use serde::{Deserialize, Deserializer};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct TestTitanicRecord {
    pub passenger_id: u32,
    pub pclass: u8,
    pub name: String,
    pub sex: String,
    #[serde(deserialize_with = "option_float_to_int")]
    pub age: u32,
    pub sib_sp: u8,
    pub parch: u8,
    pub ticket: String,
    #[serde(deserialize_with = "option_float_to_some")]
    pub fare: f32,
    #[serde(deserialize_with = "option_string_to_string_cabin")]
    pub cabin: String,
    #[serde(deserialize_with = "option_string_to_string_embark")]
    pub embarked: String,
}

fn option_float_to_some<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<f32>::deserialize(deserializer).map(
        |record| {
            if let Some(fare) = record {
                fare
            } else {
                100.0
            }
        },
    )
}

fn option_float_to_int<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<f32>::deserialize(deserializer).map(|record| {
        if let Some(age) = record {
            age as u32
        } else {
            24 as u32
        }
    })
}

fn option_string_to_string_embark<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(|record| {
        if let Some(embarked) = record {
            embarked
        } else {
            "S".to_string()
        }
    })
}

fn option_string_to_string_cabin<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(|record| {
        if let Some(cabin) = record {
            cabin
        } else {
            "C23 C25 C27".to_string()
        }
    })
}

pub struct RawTestData(Vec<TestTitanicRecord>);

impl RawTestData {
    pub fn new(path: &str) -> RawTestData {
        RawTestData(Self::collect_records(path))
    }

    pub fn get_all_rows(&self) -> &Vec<TestTitanicRecord> {
        &self.0
    }

    fn collect_records(path: &str) -> Vec<TestTitanicRecord> {
        let mut reader = csv::ReaderBuilder::new()
            .from_path(std::path::Path::new(path))
            .expect("could not read csv");

        reader
            .deserialize()
            .map(|item| item.expect("not a valid record"))
            .collect()
    }

    pub fn get_ages(&self) -> Vec<f32> {
        self.0.iter().map(|record| record.age as f32).collect()
    }

    pub fn get_classes(&self) -> Vec<f32> {
        self.0.iter().map(|record| record.pclass as f32).collect()
    }

    pub fn get_sexes(&self) -> Vec<f32> {
        self.0
            .iter()
            .map(|record| if record.sex == "male" { 1.0 } else { 0.0 })
            .collect()
    }

    pub fn get_parch(&self) -> Vec<f32> {
        self.0.iter().map(|record| record.parch as f32).collect()
    }

    pub fn get_sibsp(&self) -> Vec<f32> {
        self.0.iter().map(|record| record.sib_sp as f32).collect()
    }
}

pub struct TestBatch<B: Backend> {
    pub inputs: Tensor<B, 2>, // [[f32, f32, ...], [f32, f32, ...], ...]
    pub ids: Vec<u32>,
}

pub struct TestDataPoint<B: Backend> {
    features: Tensor<B, 1>,
    id: u32,
}

impl<B: Backend> TestDataPoint<B> {
    pub fn new(id: u32, feature: [f32; 6], device: &B::Device) -> TestDataPoint<B> {
        let features: Tensor<B, 1> = Tensor::from_floats(feature, device);
        TestDataPoint { features, id }
    }
}

pub struct TestDataSet<B: Backend>(Vec<TestDataPoint<B>>);

impl<B: Backend> TestDataSet<B> {
    pub fn new(device: &B::Device) -> TestDataSet<B> {
        let raw_data = RawTestData::new("data/test.csv");
        let mut data = Vec::new();
        for record in raw_data.get_all_rows() {
            let age = record.age as f32;
            let class = record.pclass as f32;
            let fare = record.fare;
            let sex: f32 = if record.sex == "male" { 1.0 } else { 0.0 };
            let parch = record.parch as f32;
            let sibsp = record.sib_sp as f32;

            let id = record.passenger_id;
            let feature = [age, class, fare, sex, parch, sibsp];

            data.push(TestDataPoint::new(id, feature, device));
        }

        TestDataSet(data)
    }

    pub fn batch(&self) -> TestBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut ids: Vec<u32> = Vec::new();

        for dp in &self.0 {
            inputs.push(dp.features.clone().unsqueeze());
            ids.push(dp.id);
        }

        let inputs = Tensor::cat(inputs, 0);
        TestBatch { inputs, ids }
    }
}
