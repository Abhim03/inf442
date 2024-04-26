#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


file_to_target = {
    "salary": "salary",
    "maisons": "price"
}


def normalize(train_data, test_data, col_regr, method='mean_std'):
    """
    Normalizes all the features by linear transformation *except* for the target regression column
    specified as `col_regr`.
    Two normalization methods are implemented:
      -- `mean_std` shifts by the mean and divides by the standard deviation
      -- `maxmin` shifts by the min and divides by the difference between max and min
      *Note*: mean/std/max/min are computed on the training data
    The function returns a pair normalized_train, normalized_test. For example,
    if you had `train` and `test` pandas DataFrames with the regression col stored in column `Col`, you can do

        train_norm, test_norm = normalize(train, test, 'Col')

    to get the normalized `train_norm` and `test_norm`.
    """
    # removing the class column so that it is not scaled
    no_class_train = train_data.drop(col_regr, axis=1)
    no_class_test = test_data.drop(col_regr, axis=1)

    # scaling
    normalized_train, normalized_test = None, None
    if method == 'mean_std':
        normalized_train = (no_class_train - no_class_train.mean()) / no_class_train.std()
        normalized_test = (no_class_test - no_class_train.mean()) / no_class_train.std()
    elif method == 'maxmin':
        normalized_train = (no_class_train - no_class_train.min()) / (no_class_train.max() - no_class_train.min())
        normalized_test = (no_class_test - no_class_train.min()) / (no_class_train.max() - no_class_train.min())
    else:
        raise f"Unknown method {method}"

    # gluing back the class column and returning
    return pd.concat([train_data[col_regr], normalized_train], axis=1), pd.concat([test_data[col_regr], normalized_test], axis=1)


def get_data(base_path="../csv", file_prefix="salary", feature_cols=None, target=None, norm=False):
    assert file_prefix in ["salary", "maisons"], "Unknown file"
    if target is None:
        target = file_to_target[file_prefix]
    train_path = os.path.join(base_path, f"{file_prefix}_train.csv")
    test_path = os.path.join(base_path, f"{file_prefix}_test.csv")

    train_dataset = pd.read_csv(train_path, header=0)
    summary(train_dataset)
    test_dataset = pd.read_csv(test_path, header=0)
    summary(test_dataset)

    if norm:
        train_dataset, test_dataset = normalize(train_dataset, test_dataset, target)
    # The features used to build matrix X
    if feature_cols is None:
        feature_cols = train_dataset.columns.to_list()
        feature_cols.remove(target)
    assert isinstance(feature_cols, list), "feature_cols must be a list"

    assert set(feature_cols).intersection(train_dataset.columns) == set(feature_cols),\
        f"Missing columns {set(feature_cols).difference(train_dataset.columns)}"

    X_train = train_dataset[feature_cols]
    y_train = train_dataset[target]

    X_test = test_dataset[feature_cols]
    y_test = test_dataset[target]

    return X_train, y_train, X_test, y_test


def summary(dataset):
    print(f'Shape of the data {dataset.shape}')
    print(dataset.head(5))
    print(dataset.describe())
    print('\n\n')


def fit_and_predict(X_train, y_train, X_test, y_test, regressor, verbose=False):
    assert isinstance(regressor, LinearRegression) or isinstance(regressor, KNeighborsRegressor)
    regressor.fit(X_train, y_train)

    if isinstance(regressor, LinearRegression):
        print(f'\tintercept = {regressor.intercept_}')
        print(f'\tcoefficient = {regressor.coef_}')

    y_pred = regressor.predict(X_test)
    if verbose:
        for a, b in zip(y_test, y_pred):
            print(f'  true value: {a} \t predicted value: {b}')
    return y_pred


def evaluate_performance(y_test, y_pred):
    print('\n\n')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('\n')


if __name__ == "__main__":
    # Questions 1 to 4: use the "salary" dataset
    # Questions 5 to 8: use the "maisons" dataset
    # Hint: see arguments to get_data
    X_train, y_train, X_test, y_test = get_data()

    # Questions 1 to 4: use linear regression
    # Questions 5 to 8: use kNN regression
    # Hint: see arguments to fit_and_predict
    regressor = LinearRegression()
    # regressor = KNeighborsRegressor(...)
    y_pred = fit_and_predict(X_train, y_train, X_test, y_test, regressor)
    evaluate_performance(y_test, y_pred)
