# Titanic Tabular Binary Classification

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Goal

It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.

### Metric

Your score is the percentage of passengers you correctly predict. This is known as accuracy.

### Submission File Format

You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.
The file should have exactly 2 columns:

PassengerId (sorted in any order)
Survived (contains your binary predictions: 1 for survived, 0 for deceased)

## Data

The dataset used in this challenge can be found [on kaggle](https://www.kaggle.com/competitions/titanic/data).

| Variable | Definition                                 | Key                                            |
| -------- | ------------------------------------------ | ---------------------------------------------- |
| survival | Survival                                   | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

### Cleaning

Often, raw data is a bit messy. It can be missing values or have unusual values. Outliers can also be present. So the first step is to clean the data. And to do that, we need to know where the messiness is and then how to clean it.

First, let's see what is missing.

### Visualize

Visualizing data can help with insight and weaknesses in the data. Those weaknesses need to be amended and the insights need to be leveraged.

We will use [Plotly](https://github.com/plotly/plotly.rs/tree/main) to create plots.

Here are some examples:

- Age histogram
- Fare histogram
- Survival rate by class
- Survival rate by sex
- Survival rate by age

### Correlations

Correlations can help with finding relationships between variables. Strong correlations can be indicative of a linear relationship.

- Correlation between fare and survival: 0.26
- Correlation between age and survival: 0.50

### Insights

- Most people were between 20-30 years old
- The ratio of survivors to non-survivors for each class is:
  - 1st class: 8:25
  - 2nd class: 9:10
  - 3rd class: 1.7:1
- Highest survival rate by age was below the age of 10.
- The survival rate for females was higher than for males
