use plotly::{common::Title, layout::Axis, Bar, Histogram, Layout, Plot, Scatter};
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

    // visualize data
    // age_histogram(&records);
    // fare_histogram(&records);
    // survive_by_class_bar_chart(&records);
    // survive_by_sex_bar_chart(&records);
    // survive_by_age_bar_chart(&records);

    // correlations
}

fn survive_by_class_bar_chart(records: &Vec<TitanicRecord>) {
    let mut class_survival_count = vec![(0, 0, 0); 3]; // (class, survived, not_survived)
    for record in records {
        let class = record.Pclass as usize - 1;
        if record.Survived == 1 {
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
    let ages: Vec<f32> = records.iter().filter_map(|record| record.Age).collect();

    // Define bins and count frequencies
    let bin_size = 10.0;
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
        if record.Sex == "male" {
            male_count += 1;
            if record.Survived == 1 {
                male_survived += 1;
            }
        } else if record.Sex == "female" {
            female_count += 1;
            if record.Survived == 1 {
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
    let fares: Vec<f32> = records.iter().map(|record| record.Fare).collect();

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
    let bin_size = 10.0;
    let num_bins = 8;
    let mut age_bins = vec![(0, 0); num_bins]; // (total, survived)

    for record in records {
        if let Some(age) = record.Age {
            let bin_index = (age / bin_size) as usize;
            if bin_index < age_bins.len() {
                age_bins[bin_index].0 += 1;
                if record.Survived == 1 {
                    age_bins[bin_index].1 += 1;
                }
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

fn correlation_scatter_plot(
    x_vals: &[f32],
    y_vals: &[f32],
    title: &str,
    x_label: &str,
    y_label: &str,
) {
    let coef = pearson_correlation(x_vals, y_vals);

    // Create the scatter plot
    let trace = Scatter::new(x_vals.to_vec(), y_vals.to_vec())
        .name(title)
        .mode(plotly::common::Mode::Markers);

    let layout = Layout::new()
        .title(format!("{}: r = {:.2}", title, coef))
        .x_axis(Axis::new().title(x_label))
        .y_axis(Axis::new().title(y_label).tick_values(vec![0.0, 2.0]));

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    // Save the plot to an HTML file
    plot.write_html("age_vs_fare_scatter.html");
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
