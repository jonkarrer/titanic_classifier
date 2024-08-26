use std::collections::HashMap;

use plotly::{layout::Axis, Bar, Layout, Plot, Scatter};
use serde::{Deserialize, Deserializer};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct TitanicRecord {
    passenger_id: u32,
    survived: u8,
    pclass: u8,
    name: String,
    sex: String,
    #[serde(deserialize_with = "option_float_to_int")]
    age: u32,
    sib_sp: u8,
    parch: u8,
    ticket: String,
    fare: f32,
    #[serde(deserialize_with = "option_string_to_string_cabin")]
    cabin: String,
    #[serde(deserialize_with = "option_string_to_string_embark")]
    embarked: String,
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

pub struct RecordStore {
    records: Vec<TitanicRecord>,
}

impl RecordStore {
    pub fn new() -> RecordStore {
        RecordStore {
            records: Self::collect_records(),
        }
    }

    fn collect_records() -> Vec<TitanicRecord> {
        let mut reader = csv::ReaderBuilder::new()
            .from_path(std::path::Path::new("data/train.csv"))
            .expect("could not read csv");

        reader
            .deserialize()
            .map(|item| item.expect("not a valid record"))
            .collect()
    }

    pub fn get_ages(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| record.age as f32)
            .collect()
    }

    pub fn get_survived(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| record.survived as f32)
            .collect()
    }

    pub fn get_classes(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| record.pclass as f32)
            .collect()
    }

    pub fn get_fares(&self) -> Vec<f32> {
        self.records.iter().map(|record| record.fare).collect()
    }

    pub fn get_sexes(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| if record.sex == "male" { 1.0 } else { 0.0 })
            .collect()
    }

    pub fn get_parch(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| record.parch as f32)
            .collect()
    }

    pub fn get_sibsp(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| record.sib_sp as f32)
            .collect()
    }
}

pub struct Visualizer {
    store: RecordStore,
}

impl Visualizer {
    pub fn new() -> Visualizer {
        Visualizer {
            store: RecordStore::new(),
        }
    }

    fn survive_by_class_bar_chart(records: &Vec<TitanicRecord>) {
        let mut class_survival_count = vec![(0, 0, 0); 3]; // (class, survived, not_survived)
        for record in records {
            let class = record.pclass as usize - 1;
            if record.survived == 1 {
                class_survival_count[class].1 += 1;
            } else {
                class_survival_count[class].2 += 1;
            }
        }

        // Prepare data for Plotly
        let classes: Vec<&str> = vec!["First Class", "Second Class", "Third Class"];
        let survived_counts: Vec<i32> = class_survival_count
            .iter()
            .map(|&(_, survived, _)| survived)
            .collect();
        let not_survived_counts: Vec<i32> = class_survival_count
            .iter()
            .map(|&(_, _, not_survived)| not_survived)
            .collect();

        // Create the bar chart
        let trace1 = Bar::new(classes.clone(), survived_counts).name("Survived");
        let trace2 = Bar::new(classes, not_survived_counts).name("Not Survived");

        let layout = Layout::new()
            .x_axis(Axis::new().title("Passenger Class"))
            .y_axis(Axis::new().title("Count"));

        let mut plot = Plot::new();
        plot.add_trace(trace1);
        plot.add_trace(trace2);
        plot.set_layout(layout);

        plot.write_html("survive_by_class_bar_chart.html");
    }

    fn age_histogram(records: &Vec<TitanicRecord>) {
        // Extract ages and filter out None values
        let ages: Vec<u32> = records.iter().map(|record| record.age).collect();

        // Define bins and count frequencies
        let bin_size = 10;
        let mut bins = vec![0; 10]; // Assuming ages range from 0 to 100
        for age in ages {
            let bin_index = (age / bin_size) as usize;
            if bin_index < bins.len() {
                bins[bin_index] += 1;
            }
        }

        // Prepare data for Plotly
        let bin_labels: Vec<String> = (0..bins.len())
            .map(|i| format!("{}-{}", i * 10, (i + 1) * 10))
            .collect();

        // Create the bar chart
        let trace = Bar::new(bin_labels, bins).name("Age Distribution");

        let layout = Layout::new()
            .x_axis(Axis::new().title("Age Bins"))
            .y_axis(Axis::new().title("Frequency"));

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);

        // Save the plot to an HTML file
        plot.write_html("age_histogram.html");
    }

    fn survive_by_sex_bar_chart(records: &Vec<TitanicRecord>) {
        // Calculate survival rates by sex
        let mut male_count = 0;
        let mut male_survived = 0;
        let mut female_count = 0;
        let mut female_survived = 0;

        for record in records {
            if record.sex == "male" {
                male_count += 1;
                if record.survived == 1 {
                    male_survived += 1;
                }
            } else if record.sex == "female" {
                female_count += 1;
                if record.survived == 1 {
                    female_survived += 1;
                }
            }
        }

        let male_survival_rate = (male_survived as f32 / male_count as f32) * 100.0;
        let female_survival_rate = (female_survived as f32 / female_count as f32) * 100.0;

        // Prepare data for Plotly
        let sexes = vec!["Male", "Female"];
        let survival_rates = vec![male_survival_rate, female_survival_rate];

        // Create the bar chart
        let trace = Bar::new(sexes, survival_rates).name("Survival Rate");

        let layout = Layout::new()
            .title("Survival Rate by Sex")
            .x_axis(Axis::new().title("Sex"))
            .y_axis(Axis::new().title("Survival Rate (%)"));

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);

        // Save the plot to an HTML file
        plot.write_html("survival_rate_by_sex.html");
    }

    fn fare_histogram(records: &Vec<TitanicRecord>) {
        // Extract fares
        let fares: Vec<f32> = records.iter().map(|record| record.fare).collect();

        // Define bins and count frequencies
        let bin_size = 10.0;
        let max_fare = *fares
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let num_bins = (max_fare / bin_size).ceil() as usize;
        let mut bins = vec![0; num_bins];

        for fare in fares {
            let bin_index = (fare / bin_size) as usize;
            if bin_index < bins.len() {
                bins[bin_index] += 1;
            }
        }

        // Prepare data for Plotly
        let bin_labels: Vec<String> = (0..bins.len())
            .map(|i| format!("{}-{}", i * 10, (i + 1) * 10))
            .collect();

        // Create the bar chart
        let trace = Bar::new(bin_labels, bins).name("Fare Distribution");

        let layout = Layout::new()
            .title("Fare Histogram")
            .x_axis(Axis::new().title("Fare Bins"))
            .y_axis(Axis::new().title("Frequency"));

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);

        // Save the plot to an HTML file
        plot.write_html("fare_histogram.html");
    }

    fn survive_by_age_bar_chart(records: &Vec<TitanicRecord>) {
        // Define age bins and calculate survival rates
        let bin_size = 10;
        let num_bins = 8;
        let mut age_bins = vec![(0, 0); num_bins]; // (total, survived)

        for record in records {
            let age = record.age;
            let bin_index = (age / bin_size) as usize;
            if bin_index < age_bins.len() {
                age_bins[bin_index].0 += 1;
                if record.survived == 1 {
                    age_bins[bin_index].1 += 1;
                }
            }
        }

        // Calculate survival rates
        let survival_rates: Vec<f32> = age_bins
            .iter()
            .map(|&(total, survived)| {
                if total == 0 {
                    0.0
                } else {
                    (survived as f32 / total as f32) * 100.0
                }
            })
            .collect();

        // Prepare data for Plotly
        let bin_labels: Vec<String> = (0..num_bins)
            .map(|i| format!("{}-{}", i * 10, (i + 1) * 10))
            .collect();

        // Create the bar chart
        let trace = Bar::new(bin_labels, survival_rates).name("Survival Rate");

        let layout = Layout::new()
            .title("Survival Rate by Age")
            .x_axis(Axis::new().title("Age Bins"))
            .y_axis(Axis::new().title("Survival Rate (%)"));

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);

        // Save the plot to an HTML file
        plot.write_html("survival_rate_by_age.html");
    }

    fn scatter_plot(x_vals: &[f32], y_vals: &[f32], title: &str, x_label: &str, y_label: &str) {
        let coef = Self::pearson_correlation(x_vals, y_vals);

        // Create the scatter plot
        let x = x_vals.to_vec();
        let y = y_vals.to_vec();
        dbg!(x.len(), y.len());
        let trace = Scatter::new(x, y)
            .name(title)
            .mode(plotly::common::Mode::Markers);

        let layout = Layout::new()
            .title(format!("{}: r = {:.2}", title, coef))
            .x_axis(Axis::new().title(x_label))
            .y_axis(Axis::new().title(y_label));

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);

        // Save the plot to an HTML file

        plot.write_html(format!(
            "{}_{}_scatter.html",
            x_label.to_lowercase(),
            y_label.to_lowercase()
        ));
    }

    fn mode_nums(numbers: &Vec<u32>) -> Vec<u32> {
        let mut map = HashMap::new();
        for integer in numbers {
            let count = map.entry(integer).or_insert(0);
            *count += 1;
        }

        let max_value = map.values().cloned().max().unwrap_or(0);

        map.into_iter()
            .filter(|&(_, v)| v == max_value)
            .map(|(&k, _)| k)
            .collect()
    }

    fn mode_strs(strings: Vec<String>) -> Vec<String> {
        let mut map = HashMap::new();
        for string in strings {
            let count = map.entry(string).or_insert(0);
            *count += 1;
        }

        let max_value = map.values().cloned().max().unwrap_or(0);

        map.into_iter()
            .filter(|&(_, v)| v == max_value)
            .map(|(k, _)| k)
            .collect()
    }

    fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() {
            panic!("Vectors must have the same length");
        }

        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let mut covariance = 0.0;
        let mut variance_x = 0.0;
        let mut variance_y = 0.0;

        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            covariance += diff_x * diff_y;
            variance_x += diff_x * diff_x;
            variance_y += diff_y * diff_y;
        }

        let stddev_x = variance_x.sqrt();
        let stddev_y = variance_y.sqrt();

        covariance / (stddev_x * stddev_y)
    }
}
