import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
# from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), \
                                           generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    adaBoost.fit(train_X, train_y)
    fig = go.Figure()
    train_partial_loss = []
    test_partial_loss = []
    for i in range(1, n_learners + 1):
        train_partial_loss.append(adaBoost.partial_loss(train_X, train_y,  i))
        test_partial_loss.append(adaBoost.partial_loss(test_X, test_y, i))
    fig.add_trace(go.Scatter(x=[i for i in range(1, n_learners + 1)],
                  y=train_partial_loss, name="Train error"))
    fig.add_trace(go.Scatter(x=[i for i in range(1, n_learners + 1)],
                            y=test_partial_loss, name="Test error"))
    fig.update_layout(title="The training and test errors as a function of "
                            f"the number of fitted learners, noise = {noise}",
                      xaxis_title="Number of fitted learners",
                      yaxis_title="The error rate")
    fig.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                 np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    graphs_titles = [f"n = {5}", f"n = {50}", f"n = {100}", f"n = {250}"]
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=graphs_titles,
                          horizontal_spacing=0.05, vertical_spacing=0.1)

    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x, t),
                                          lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                               mode="markers", showlegend=False,
                               marker=dict(color=test_y,
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color="black", width=1)))],
                       rows=(i//2) + 1, cols=(i % 2) + 1)
    fig2.update_layout(title="decision boundary obtained by using the the "
                             "ensemble up to iteration 5, 50, 100 and 250, "
                             f"noise={noise}")
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    minimal_ensemble_size = np.argmin(test_partial_loss) + 1
    minimal_ensemble_accuracy = accuracy(y_true=test_y,
                                     y_pred=adaBoost.partial_predict(test_X,
                                                        minimal_ensemble_size))
    fig3 = go.Figure()
    fig3.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x,
                                    minimal_ensemble_size),
                                    lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                               mode="markers", showlegend=False,
                               marker=dict(color=test_y,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])
    fig3.update_layout(title=f"Ensemble size with the lowest test error "
                             f"is {minimal_ensemble_size} with "
                             f"accuracy of {minimal_ensemble_accuracy} and "
                             f"noise = {noise}",
                       xaxis_title="feature 1", yaxis_title="feature 2")
    fig3.show()

    # Question 4: Decision surface with weighted samples
    D = adaBoost.D_ / np.max(adaBoost.D_) * 5
    fig4 = go.Figure()
    fig4.add_traces([decision_surface(adaBoost.predict,
                                    lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                               mode="markers", showlegend=False,
                               marker=dict(size=D, color=train_y,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])
    fig4.update_layout(title="Training set with a point size proportional"
                             f" to it’s weight, noise={noise}",
                       xaxis_title="feature 1", yaxis_title="feature 2")
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
