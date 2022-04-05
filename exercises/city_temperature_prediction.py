import random

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def country_to_columns(full_data):
    return pd.get_dummies(full_data["Country"])


def city_to_columns(full_data):
    return pd.get_dummies(full_data["City"])


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).\
        dropna().drop_duplicates()
    full_data = full_data[full_data["Temp"] > -72]
    day_of_year = []
    for date in full_data["Date"]:
        day_of_year.append(pd.Period(date, "D").day_of_year)
    # cities = city_to_columns(full_data)
    # countries = country_to_columns(full_data)
    full_data["day_of_year"] = day_of_year
    features = full_data[["Country", "City", "day_of_year",
                          "Year", "Month", "Day"]]
    # features = pd.concat([features, cities], axis=1)
    # features = pd.concat([features, countries], axis=1)
    labels = full_data["Temp"]
    return features, labels


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df, responses = load_data(
        "C:\\Users\\idoro\\Desktop\\IML\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_without_temp = df[df["Country"] == "Israel"]
    features_with_temp = pd.concat([df, responses], axis=1)
    israel_with_temp = \
        features_with_temp[features_with_temp["Country"] == "Israel"]
    israel_temp = responses.reindex_like(israel_with_temp)
    fig = px.scatter(pd.DataFrame({"x": israel_with_temp["day_of_year"],
                     "y": israel_temp}), x="x", y="y",
                    labels={"x": "Day of year", "y": "Temperature"},
                    title="The temperature as a function of the day of year",
                    color=israel_with_temp["Year"].astype(str))
    fig.show()

    israel_by_month = israel_with_temp.groupby("Month").agg({"Temp": "std"})
    months = (israel_with_temp["Month"].drop_duplicates()).values
    fig2 = px.bar(israel_by_month, x=months, y="Temp",
                  labels={"x": "Month", "Temp": "Standard Deviation"},
                  title="The standard deviation of the daily temperatures "
                        "as a function of months")
    fig2.show()

    # Question 3 - Exploring differences between countries
    grouped_by_country = features_with_temp.groupby(["Country", "Month"])
    country_month_mean = grouped_by_country.mean().reset_index()
    country_month_std = grouped_by_country.std().reset_index()
    country_month_mean.insert(1, "std", country_month_std["Temp"])
    fig3 = px.line(country_month_mean, x="Month", y="Temp", error_y="std",
                   color="Country")
    fig3.update_layout(title="The average and standard deviation as a "
                             "function of Country and Month",
                       xaxis_title="Month",
                       yaxis_title="Average month temperature")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    israel_features_train, israel_temp_train, israel_features_test, \
        israel_temp_test = split_train_test(israel_without_temp,
                            israel_temp, train_proportion=0.75)
    israel_losses = []
    for k in range(1, 11):
        poly_estimator = PolynomialFitting(k)
        poly_estimator.fit((israel_features_train["day_of_year"]).to_numpy(),
                           israel_temp_train)
        rounded_loss = np.round(poly_estimator.loss
                             (israel_features_test["day_of_year"].to_numpy(),
                            israel_temp_test), 2)
        israel_losses.append(rounded_loss)
    fig4 = px.bar(x=[i for i in range(1, 11)], y=israel_losses)
    fig4.update_layout(title="The test error of the model as a function of "
                       "the polynomial degree",
                       xaxis_title="Polynomial Degree",
                       yaxis_title="Test Error")
    fig4.show()
    print(israel_losses)

    # Question 5 - Evaluating fitted model on different countries
    min_k = np.argmin(israel_losses) + 1
    israel_poly = PolynomialFitting(min_k)
    israel_poly.fit(israel_without_temp["day_of_year"].to_numpy(),
                    israel_temp)
    losses_by_countries = {}
    countries = set(features_with_temp["Country"])
    for country in countries:
        if country == "Israel":
            continue
        features_by_country = df[df["Country"] == country]
        temp_of_country = responses.reindex_like(features_by_country)
        rounded_loss = np.round(israel_poly.loss(
            features_by_country["day_of_year"].to_numpy(),
            temp_of_country), 2)
        losses_by_countries[country] = rounded_loss
    fig5 = px.bar(x=losses_by_countries.keys(),
                  y=losses_by_countries.values(),
                  color=losses_by_countries.keys())
    fig5.update_layout(title="The test error of the model fitted for Israel "
                             "as a function of the other countries"
                       "the polynomial degree",
                       xaxis_title="Country",
                       yaxis_title="Test Error")
    fig5.show()

