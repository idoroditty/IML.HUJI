from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def floors_num_to_columns(full_data):
    return pd.get_dummies(full_data["floors"])


def zipcode_to_columns(full_data):
    full_data["zipcode_by_area"] = full_data["zipcode"].apply(lambda x:
                                                      str(np.round(x / 10)))
    return pd.get_dummies(full_data["zipcode_by_area"])


def year_to_duration(full_data):
    year_exist = np.abs(full_data["yr_built"] - 2022)
    year_since_renovation = np.abs(full_data["yr_renovated"] - 2022)
    last_renovated_or_build_year = pd.concat([year_exist,
                                              year_since_renovation],
                                             axis=1).min(axis=1)
    full_data.drop("yr_built", axis=1)
    full_data.insert(14, "yr_exist", year_exist)
    full_data.insert(16, "year_since_renovation", year_since_renovation)
    full_data.insert(18, "last_changed",
                     last_renovated_or_build_year)


def date_to_year(full_data):
    dates = full_data["date"]
    selling_year = []
    for date in dates:
        temp_year = pd.to_datetime(date).year
        selling_year.append(temp_year)
    full_data.insert(1, "selling_year", selling_year)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    # corr_data = pd.DataFrame(np.round(full_data.corr(), 3))
    # corr_fig = px.imshow(corr_data, text_auto=True, height=1000, width=1000)
    # corr_fig.show()
    full_data.drop(full_data[(full_data["id"] == 0)].index, inplace=True)
    floors_by_categories = floors_num_to_columns(full_data)
    year_to_duration(full_data)
    date_to_year(full_data)
    zipcode_by_categories = zipcode_to_columns(full_data)
    features = full_data[["bedrooms",
                          "bathrooms",
                          "sqft_living",
                          "sqft_lot",
                          "condition",
                          "view",
                          "grade",
                          "sqft_above",
                          "last_changed"]]
    features = pd.concat([features, pd.get_dummies(full_data["selling_year"])],
                         axis=1)
    features = pd.concat([features, floors_by_categories], axis=1)
    features = pd.concat([features, zipcode_by_categories], axis=1)
    labels = full_data["price"]
    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for x in X.columns:
        feature = X[x]
        corr = np.cov(feature, y)[0, 1] / np.sqrt(np.var(feature) * np.var(y))
        corr = np.round(corr, 3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feature, y=y, mode="markers"))
        fig.update_layout(title=f"Evaluation of {x} and the response.\n"
                                f"\nPearson Correlation value: {corr}",
                          xaxis_title=f"{x} values", yaxis_title="house price")
        fig.write_image(output_path+f"{x}.png", format="png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, responses = load_data(
        "C:\\Users\\idoro\\Desktop\\IML\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, responses,
                   "C:\\Users\\idoro\\Desktop\\IML\\exercises\\EX2-Q2-Graphs"
                   "\\")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, responses,
                                                        train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lin_reg = LinearRegression()
    mean_results = []
    var_results = []
    percents = []
    for i in range(10, 101):
        current_losses = []
        for j in range(10):
            new_train_X, new_train_y, temp_test_X, temp_test_y = \
                split_train_test(train_X, train_y,
                                 train_proportion=(i / 100))
            lin_reg.fit(new_train_X.values, new_train_y.values)
            current_losses.append(lin_reg.loss(test_X.values,
                                             test_y.values))
        percents.append(i)
        mean_results.append(np.mean(current_losses))
        var_results.append(np.std(current_losses))
    mean_results = np.array(mean_results)
    var_results = np.array(var_results)
    fig2 = go.Figure([go.Scatter(x=percents, y=mean_results,
                                 mode="markers+lines",
                                 name="average loss",
                                 line=dict(dash="dash"),
                                 marker=dict(color="green")),
                      go.Scatter(x=percents,
                                 y=(mean_results - 2 * var_results),
                                 fill='tonexty',
                                 mode="lines",
                                 line=dict(color="lightgrey"),
                                 showlegend=False),
                      go.Scatter(x=percents,
                                 y=(mean_results + 2 * var_results),
                                 fill='tonexty',
                                 mode="lines",
                                 line=dict(color="lightgrey"),
                                 showlegend=False)])
    fig2.update_layout(title="Mean loss as a function of training by percent",
                       xaxis_title="Training percent",
                       yaxis_title="Mean loss")
    fig2.show()

    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011,
                       345856.57131275, 563867.1347574, 395102.94362135])
    print(np.round(mean_square_error(y_true, y_pred), 3))
