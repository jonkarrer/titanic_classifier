// use plotly::{Plot, Scatter};

use std::fs::File;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TitanicRecord {
    PassengerId: u32,
    Survived: u8,
    Pclass: u8,
    Name: String,
    Sex: String,
    Age: Option<f32>,
    SibSp: u8,
    Parch: u8,
    Ticket: String,
    Fare: f32,
    Cabin: Option<String>,
    Embarked: Option<String>,
}

fn main() {
    // ** 1: Prep Data **
    // raw csv to vector of records
    let mut reader = csv::ReaderBuilder::new()
        .from_path(std::path::Path::new("data/train.csv"))
        .expect("could not read csv");

    let records: Vec<TitanicRecord> = reader
        .deserialize()
        .map(|item| item.expect("not a valid record"))
        .collect();

    dbg!(&records[0]);

    // dbg!(df);
    // let mut plot = Plot::new();
    // let trace = Scatter::new(vec![0, 1, 2], vec![2, 1, 0]);
    // plot.add_trace(trace);

    // plot.write_html("out.html");
}
