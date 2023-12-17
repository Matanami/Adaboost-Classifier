# Adaboost-Classifier
## Introduction

This Python script implements the Adaboost algorithm for binary classification. Adaboost is an ensemble learning technique that combines weak learners to create a strong classifier. The script trains and evaluates an Adaboost classifier on a given dataset, visualizing the training and test errors over iterations.

## Dependencies

Make sure to install the following dependencies before running the script:

```bash
pip install matplotlib numpy
```

## Usage

To run the script, execute the following command:

```bash
python adaboost_classifier.py
```

## Code Overview

run_adaboost(X_train, y_train, X_test, y_test, T): Runs the Adaboost algorithm for a specified number of iterations T.
WL(D_t, X_train, y_train): Implements the Weak Learner algorithm to find the best threshold, polarity, and feature index.
calculate_zero_one_error(predictions, labels, weights=None): Calculates the zero-one error for given predictions and labels.
classify(X, threshold, polarity, index): Classifies input data based on a specified threshold, polarity, and feature index.
strong_classify(X, hypotheses, alpha_vals): Classifies input data using the strong classifier built from Adaboost.
main(): Main function to load data, run Adaboost, and visualize results.
## Results

The script produces visualizations of training and test errors over iterations, showcasing the performance of the Adaboost algorithm.

a(errors_train, errors_test): Plots the training and test errors over iterations.
c(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T): Plots the exponential loss on training and test data over iterations.
The resulting plots are saved as "a.png" and "c.png."