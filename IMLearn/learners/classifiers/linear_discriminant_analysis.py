from typing import NoReturn

# from . import GaussianNaiveBayes
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.zeros(self.classes_.shape[0])
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            X_as_class = X[y == self.classes_[i]]
            class_var_count = np.count_nonzero(y == self.classes_[i])
            self.pi_[i] = class_var_count / y.shape[0]
            self.mu_[i] = X_as_class.mean(axis=0)
            diff = X_as_class - self.mu_[i]
            self.cov_ += (diff.transpose() @ diff)
        self.cov_ /= (X.shape[0] - self.classes_[0])
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        responses = []
        for x in X:
            responses.append(self.classes_[self.get_maximal_class(x)])
        responses = np.array(responses)
        return responses

    def get_maximal_class(self, current_x):
        predictions = []
        for k in range(self.classes_.shape[0]):
            ak = self._cov_inv @ self.mu_[k]
            bk = np.log(self.pi_[k]) - (1 / 2) * \
                (self.mu_[k] @ self._cov_inv @ self.mu_[k])
            predictions.append(((ak.transpose() @ current_x) + bk))
        predictions = np.array(predictions)
        return np.argmax(predictions)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = []
        for k in range(self.classes_.shape[0]):
            likelihoods.append(self.get_class_likelihood(X, k))
        return np.array(likelihoods)

    def get_class_likelihood(self, X, current_class):
        d = X.shape[1]
        power_exp = (X - self.mu_[current_class]) @ self._cov_inv * \
                    (X - self.mu_[current_class])
        sum_power_exp = np.sum(power_exp, axis=1)
        sum_power_exp *= -0.5
        det_val = det(self.cov_)
        likelihood = (1 / (np.sqrt(np.power(2 * np.pi, d) *
                            det_val))) * np.exp(sum_power_exp)
        likelihood *= self.pi_[current_class]
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        predicted_y = self._predict(X)
        return misclassification_error(y, predicted_y)

