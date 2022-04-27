from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            X_as_class = X[y == self.classes_[i]]
            class_var_count = np.count_nonzero(y == self.classes_[i])
            self.pi_[i] = class_var_count / y.shape[0]
            self.mu_[i] = X_as_class.mean(axis=0)
            self.vars_[i] = X_as_class.var(axis=0, ddof=1)

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
        likelihoods = self.likelihood(X)
        return np.array([self.classes_[np.argmax(likelihood)]
                        for likelihood in likelihoods])

    def calculate_likelihood(self, current_x):
        temp = []
        for k in range(self.classes_.shape[0]):
            pdf_log_sum = np.sum(np.log(self.calculate_pdf(current_x, k)))
            pdf_log_sum += np.log(self.pi_[k])
            temp.append(pdf_log_sum)
        return np.array(temp)

    def calculate_pdf(self, current_x, current_class):
        sqrt = np.sqrt(2 * np.pi * self.vars_[current_class])
        exp_pow = - np.power((current_x - self.mu_[current_class]), 2)
        exp_pow /= (2 * self.vars_[current_class])
        pdf = (1 / sqrt) * np.exp(exp_pow)
        return pdf

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
        return np.array([self.calculate_likelihood(x) for x in X])

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
