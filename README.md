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

Often, raw data is a bit messy. Let's clean it up.

#### Missing data

We see age is missing 177, cabin is missing 687, and embarked is missing 2.

The mode for age is 24, the mode for cabin is "C23 C25 C27", and the mode for embarked is "S".

Simple enough, we can fill in the missing values.

#### Outliers

There are 3 fare outliers, we will just remove them.

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

- Correlation between age and survival: -0.05
- Correlation between fare and survival: 0.26
- Correlation between class and survival: -0.37
- Correlation between sex and survival: -0.54
- Correlation between parch and survival: 0.08
- Correlation between sibsp and survival: -0.04

### Insights

- Most people were between 20-30 years old
- The ratio of survivors to non-survivors for each class is:
  - 1st class: 8:25
  - 2nd class: 9:10
  - 3rd class: 1.7:1
- Highest survival rate by age was below the age of 10.
- The survival rate for females was higher than for males
- Weak correlation between survival and sex, class, and age
- Strong correlation between survival and fare

## Classification

Now we need to create our features and classifications.

The fields that we are interested in are:

- sex
- age
- sibsp
- fare
- class

These will be our features. The **survival** field will be our classification.

### Data Points

A single data point for this will be a 1D tensor of length 6 for the features, and a 1D tensor of length 1 for the classification. We group this into a struct called `DataPoint`. Simply looping through the data set and creating the data points will do the trick.

### Batch

We need to create 2 tensors for our model to interpret the data. Our feature tensor will be a 2D tensor, with shape [798, 6]. That's 798 rows and 6 columns. Our target tensor will be a 2D tensor, with shape [798, 1]. This will be considered a batch that the model can interpret. Since we are dealing with little data, we can just create a single batch.

## Model

Now we need to create the model to send our batches to. A simple regression neural network has these characteristics:

- Input Layer: A Linear layer with 6 inputs (feature size), and 64 outputs (hidden size)
- Activation: A Relu activation layer that will be applied to the hidden layer so that the output will be positive
- Output Layer: A Linear layer with 1 input (hidden size), and 1 output (target size)

### Loss Function

We will simply forward the batches through the model's layers, and calculate the loss. For this task, we will use the Mean Squared Error (MSE) loss function.

### Optimizer

Now we need to choose the optimizer and learning rate. For this we will use Stochastic Gradient Descent (SGD) with a learning rate of 1e-4 to start.

## Training

Now we need to bring it altogether and train our model. We loop through epochs, and on each epoch we pass the batch through the model, calculate the loss, and backpropagate the gradients. Repeat until trained.

## Results

### Run one

Configuration:

- Loss Function: Mean Squared Error
- Optimizer: Stochastic Gradient Descent
- Learning Rate: 1e-4
- Epochs: 10
- Batch Size: 798
- Features: 6
- Hidden Size: 64
- Target: 1
- Activation: Relu

```shell
[Train - Epoch 0] Loss 31.089 | Accuracy 38.221 %
*** [Validate - Epoch 0] Loss 5.670 | Accuracy 37.778 %
[Train - Epoch 1] Loss 7.438 | Accuracy 38.221 %
*** [Validate - Epoch 1] Loss 2.156 | Accuracy 37.778 %
[Train - Epoch 2] Loss 2.521 | Accuracy 38.221 %
*** [Validate - Epoch 2] Loss 1.223 | Accuracy 37.778 %
[Train - Epoch 3] Loss 1.389 | Accuracy 38.221 %
*** [Validate - Epoch 3] Loss 0.906 | Accuracy 37.778 %
[Train - Epoch 4] Loss 1.050 | Accuracy 36.341 %
*** [Validate - Epoch 4] Loss 0.756 | Accuracy 34.444 %
[Train - Epoch 5] Loss 0.892 | Accuracy 35.464 %
*** [Validate - Epoch 5] Loss 0.664 | Accuracy 32.222 %
[Train - Epoch 6] Loss 0.788 | Accuracy 34.461 %
*** [Validate - Epoch 6] Loss 0.597 | Accuracy 34.444 %
[Train - Epoch 7] Loss 0.707 | Accuracy 35.088 %
*** [Validate - Epoch 7] Loss 0.546 | Accuracy 34.444 %
[Train - Epoch 8] Loss 0.641 | Accuracy 35.213 %
*** [Validate - Epoch 8] Loss 0.504 | Accuracy 34.444 %
[Train - Epoch 9] Loss 0.588 | Accuracy 35.088 %
*** [Validate - Epoch 9] Loss 0.471 | Accuracy 34.444 %
[Train - Epoch 10] Loss 0.544 | Accuracy 35.464 %
```
