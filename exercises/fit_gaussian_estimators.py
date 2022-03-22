from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    true_mu = 10
    true_sigma = 1
    number_of_samples = 1000
    normal_dist = np.random.normal(true_mu, true_sigma, number_of_samples)
    normal_dist_var = UnivariateGaussian()
    normal_dist_var.fit(normal_dist)
    print(f"({normal_dist_var.mu_},{normal_dist_var.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    normal_dist_arr = np.array(normal_dist)
    estimated_mean = []
    ms = np.linspace(10, 1000, 100)
    for i in range(10, 1001, 10):
        normal_dist_var.fit(normal_dist_arr[:i])
        estimated_mean.append(np.abs(true_mu - normal_dist_var.mu_))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=estimated_mean, name="Graph"))
    fig.update_layout(title="Expectation error as a function of the sample "
                            "size", xaxis_title="Samples Number",
                      yaxis_title="Expectation error")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    estimated_normal_dist_arr = normal_dist_var.pdf(normal_dist)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=normal_dist, y=estimated_normal_dist_arr,
                              name="Graph", mode="markers"))
    fig2.update_layout(title="PDF values as a function of the sample value",
                       xaxis_title="Sample Value", yaxis_title="Gaussian "
                                                               "Value")
    fig2.show()
    print(normal_dist_var.log_likelihood(true_mu, true_sigma, normal_dist))
    sample = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1,
                       1,  3, 2, -1, -3, 1, -4, 1, 2, 1, -4, -4, 1, 3, 2, 6,
                       -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0,
                       3, 5, 0, -2])
    print(normal_dist_var.log_likelihood(1, 1, sample))
    print(normal_dist_var.log_likelihood(10, 1, sample))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    true_mu = np.array([0, 0, 4, 0])
    true_cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                [0.5, 0, 0, 1]])
    multi_normal_dist = np.random.multivariate_normal(true_mu, true_cov, 1000)
    multi_normal_dist_var = MultivariateGaussian()
    multi_normal_dist_var.fit(multi_normal_dist)
    print(multi_normal_dist_var.mu_)
    print(multi_normal_dist_var.cov_)

    # Question 5 - Likelihood evaluation
    samples = np.linspace(-10, 10, 200)
    estimated_loglikelihood = np.zeros((samples.size, samples.size))
    for f1_ind in range(samples.size):
        f1 = samples[f1_ind]
        for f3_ind in range(samples.size):
            f3 = samples[f3_ind]
            mu = np.array([f1, 0, f3, 0]).transpose()
            estimated_loglikelihood[f1_ind, f3_ind] =  \
                multi_normal_dist_var.log_likelihood(mu, true_cov,
                                                 multi_normal_dist)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=samples, y=samples, z=estimated_loglikelihood))
    fig.update_layout(title="Heatmap of log-likelihood as a function of f1 "
                    "and f3 values", xaxis_title="f3 Values",
                    yaxis_title="f1 Values")
    fig.show()

    # Question 6 - Maximum likelihood
    max_f1_ind, max_f3_ind = 0, 0
    max_loglikelihood = -np.inf
    for f1_ind in range(samples.size):
        for f3_ind in range(samples.size):
            if estimated_loglikelihood[f1_ind, f3_ind] > max_loglikelihood:
                max_f1_ind = f1_ind
                max_f3_ind = f3_ind
                max_loglikelihood = estimated_loglikelihood[f1_ind, f3_ind]
    print(samples[max_f1_ind], samples[max_f3_ind])
    print(estimated_loglikelihood[max_f1_ind, max_f3_ind])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

