"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest

# (you will probably need to import more things here)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import numpy as np
from regression.logreg import LogisticRegressor
from regression.utils import loadDataset


def test_prediction():
    # Load data
    X_train, X_test, y_train, y_test = loadDataset(split_percent=0.8)
    
	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create models
    my_model = LogisticRegressor(
        num_feats=X_train.shape[1],
        tol=0.0001,
        max_iter=10000,
        learning_rate=0.001,
        batch_size=40,
    )  # default params
    sk_model = LogisticRegression(tol=0.00001, max_iter=1000)  # adjust to match my_model

    # Train models
    my_model.train_model(X_train, y_train, X_test, y_test)
    sk_model.fit(X_train, y_train)

    # Make predictions
    sk_pred = sk_model.predict(X_test)
    X_test = np.hstack(
        [X_test, np.ones((X_test.shape[0], 1))]
    )  # padding data with vector of ones for bias term
    my_pred = np.round(my_model.make_prediction(X_test))

    # NOTE: This sometimes doesn't pass due to some model fittings for my model not working well at all. I am not sure exactly how to address this.
    assert np.sum(sk_pred == my_pred) / len(sk_pred) > 0.5  # check that predictions are similar

    # Check inputs
    with pytest.raises(ValueError):
        my_model.make_prediction(np.array([1]))  # wrong shape
    with pytest.raises(ValueError):
        my_model.make_prediction(3)  # wrong type


def test_loss_function():
    model = LogisticRegressor(num_feats=3)

    # Check inputs
    with pytest.raises(ValueError):
        model.loss_function("1", 1)  # wrong type
    with pytest.raises(ValueError):
        model.loss_function(np.array([1, 0, 1]), "1")  # wrong type
    with pytest.raises(ValueError):
        model.loss_function(np.array([1, 0, 1]), np.array([1, 0]))  # wrong shape
    with pytest.raises(ValueError):
        model.loss_function(
            np.array([0.5, 0.2, 1]), np.array([0.5, 0.2, 1])
        )  # labels incorrect
    with pytest.raises(ValueError):
        model.loss_function(
            np.array([1, 0, 1]), np.array([-0.5, 0.2, 1])
        )  # predictions incorrect

    # Create dummy data
    y = np.array([1, 0, 1])  # dummy true values
    y_pred = np.array([0.9, 0.1, 0.9])  # dummy predictions

    assert np.allclose(log_loss(y, y_pred), model.loss_function(y, y_pred))


def test_gradient():
    # Check inputs
    model = LogisticRegressor(num_feats=3)
    with pytest.raises(ValueError):
        model.calculate_gradient("1", 1)  # wrong type

    model.W = np.array([1, 1, 1])  # dummy weights (just for testing)

    # Create dummy data
    y = np.array([1, 0, 1])  # dummy true
    y_pred = np.array(
        [0.8175744761937223, 0.9975273768433705, 0.9990889488056016]
    )  # sigmoid of model.W.dot(X) (used online calculator https://www.tinkershop.net/ml/sigmoid_calculator.html)
    X = np.array([[0, 1, 0.5], [1, 2, 3], [0, 3, 4]])  # dummy features

    # Manual gradient calculation
    manual_grad = np.dot(X.T, y_pred - y) / len(y)
    assert np.allclose(manual_grad, model.calculate_gradient(y, X))


def test_training():
    # Load data
    X_train, X_test, y_train, y_test = loadDataset(split_percent=0.8)
    
	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    X_pred = np.hstack(
        [X_test, np.ones((X_test.shape[0], 1))]
    )  # padding data with vector of ones for bias term

    # Create model
    model = LogisticRegressor(num_feats=X_train.shape[1])

    # Save checkpoint
    predictions_before = model.make_prediction(X_pred)
    weights_before = model.W.copy()

    # Train model
    model.train_model(X_train, y_train, X_test, y_test)

    predictions_after = model.make_prediction(X_pred)

    # Check that model has been trained
    assert not np.allclose(weights_before, model.W)  # check that weights have changed
    assert not np.allclose(
        predictions_before, predictions_after
    )  # check that predictions have changed
    assert len(model.loss_hist_train) > 0  # check that training has been run
    assert len(model.loss_hist_val) > 0  # check that validation has been run
    assert len(model.loss_hist_train) == len(
        model.loss_hist_val
    )  # check that training and validation have same length

    # Check that loss has decreased
    loss_before = model.loss_function(y_test, predictions_before)
    loss_after = model.loss_function(y_test, predictions_after)

    assert loss_after < loss_before
