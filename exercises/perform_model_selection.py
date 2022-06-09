from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    mu = 0
    eps = np.random.normal(mu, noise, n_samples)
    f_x = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y_without_noise = f_x(X)
    y_with_noise = y_without_noise + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y_with_noise), 2 / 3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_without_noise, mode="markers", name="Polynom"))
    fig.add_trace(go.Scatter(x=train_X[0], y=train_y[0], mode="markers", marker=dict(color="Red",
                             colorscale=[custom[0], custom[-1]]), name="Train Set"))
    fig.add_trace(go.Scatter(x=test_X[0], y=test_y[0], mode="markers", marker=dict(color="Green",
                             colorscale=[custom[0], custom[-1]]), name="Test Set"))
    fig.update_layout(title="Training and Validation score as a function of polynomial degree."
                            f" Noise={noise}, Number of samples={n_samples}",
                      xaxis_title="Polynomial Degree", yaxis_title="Score")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = []
    validation_scores = []
    temp_train_X = np.array(train_X).flatten()
    temp_train_y = np.array(train_y).flatten()
    for i in range(11):
        train_score, validation_score = cross_validate(PolynomialFitting(i), temp_train_X,
                                                       temp_train_y, mean_square_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[i for i in range(11)], y=train_scores, name="Train Scores"))
    fig2.add_trace(go.Scatter(x=[i for i in range(11)], y=validation_scores, name="Validation Scores"))
    fig2.update_layout(title="Average Training and Validation error as a function of polynomial degree."
                             f" Noise={noise}, Number of samples={n_samples}",
                      xaxis_title="Polynomial Degree", yaxis_title="Average error")
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_scores)
    polynomial_fitting = PolynomialFitting(k_star)
    polynomial_fitting.fit(np.array(train_X), np.array(train_y))
    pred = polynomial_fitting.predict(np.array(test_X))
    print("best polynomial degree: ", k_star)
    print("The test error: ", np.round(mean_square_error(np.array(test_y), pred), 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.01, 2, n_evaluations)
    ridge_train_scores = []
    ridge_validation_scores = []
    lasso_train_scores = []
    lasso_validation_scores = []
    for lam in lambdas:
        train_score, validation_score = cross_validate(RidgeRegression(lam), train_X,
                                                       train_y, mean_square_error)
        ridge_train_scores.append(train_score)
        ridge_validation_scores.append(validation_score)
        train_score, validation_score = cross_validate(Lasso(lam), train_X,
                                                       train_y, mean_square_error)
        lasso_train_scores.append(train_score)
        lasso_validation_scores.append(validation_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambdas, y=ridge_train_scores, marker=dict(color="Red",
                             colorscale=[custom[0], custom[-1]]), name="Ridge Train Set"))
    fig.add_trace(go.Scatter(x=lambdas, y=ridge_validation_scores, marker=dict(color="Blue",
                             colorscale=[custom[0], custom[-1]]), name="Ridge Validation Set"))
    fig.add_trace(go.Scatter(x=lambdas, y=lasso_train_scores, marker=dict(color="Purple",
                             colorscale=[custom[0], custom[-1]]), name="Lasso Train Set"))
    fig.add_trace(go.Scatter(x=lambdas, y=lasso_validation_scores, marker=dict(color="Green",
                            colorscale=[custom[0], custom[-1]]), name="Lasso Validation Set"))
    fig.update_layout(title="Average Training and Validation errors as a function of the "
                            "regularization parameter lambda", xaxis_title="Regularization Parameter Value",
                      yaxis_title="Average error")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_min_ind = np.argmin(np.array(ridge_validation_scores))
    lasso_min_ind = np.argmin(np.array(lasso_validation_scores))
    lam_ridge_min = lambdas[ridge_min_ind]
    print("Ridge lambda: ", lam_ridge_min)
    lam_lasso_min = lambdas[lasso_min_ind]
    print("Lasso lambda: ", lam_lasso_min)
    ridge_estimator = RidgeRegression(lam_ridge_min)
    lasso_estimator = Lasso(lam_lasso_min)
    linear_regression_estimator = LinearRegression()
    ridge_estimator.fit(train_X, train_y)
    lasso_estimator.fit(train_X, train_y)
    linear_regression_estimator.fit(train_X, train_y)
    ridge_pred = ridge_estimator.predict(test_X)
    lasso_pred = lasso_estimator.predict(test_X)
    linear_regression_pred = linear_regression_estimator.predict(test_X)
    print("Ridge test error with best lambda: ", np.round(mean_square_error(np.array(test_y), ridge_pred), 2))
    print("Lasso test error with best lambda: ", np.round(mean_square_error(np.array(test_y), lasso_pred), 2))
    print("Linear Regression test error with best lambda: ", np.round(mean_square_error(np.array(test_y),
                                                linear_regression_pred), 2))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
