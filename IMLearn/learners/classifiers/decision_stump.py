from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
# from ...base import BaseEstimator
import numpy as np
from itertools import product

from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        minimal_error_threshold = 1
        for i in range(X.shape[1]):
            pos_threshold, pos_error = self._find_threshold(X[:, i], y, 1)
            neg_threshold, neg_error = self._find_threshold(X[:, i], y, -1)
            if pos_error < neg_error:
                if pos_error < minimal_error_threshold:
                    self.threshold_ = pos_threshold
                    self.j_ = i
                    minimal_error_threshold = pos_error
                    self.sign_ = 1
            else:
                if neg_error < minimal_error_threshold:
                    self.threshold_ = neg_threshold
                    self.j_ = i
                    minimal_error_threshold = neg_error
                    self.sign_ = -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        prediction = np.array([-self.sign_ if curr[self.j_] < self.threshold_
                        else self.sign_ for curr in X])
        return prediction

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_values = np.sort(values)
        sorted_values_indexes = np.argsort(values)
        new_labels = np.take(labels, sorted_values_indexes)
        min_threshold = 0
        min_threshold_error = 1
        temp_labels = np.ones(values.shape[0])
        temp_labels *= sign
        for i in range(sorted_values.shape[0]):
            temp_error = np.sum(np.where(temp_labels != np.sign(new_labels),
                                         np.abs(new_labels), 0)) / len(sorted_values)
            if temp_error < min_threshold_error:
                min_threshold_error = temp_error
                min_threshold = sorted_values[i]
            temp_labels[i] = -sign
        return min_threshold, min_threshold_error

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
        predicted_y = self._predict(X)
        return np.sum(np.where(predicted_y != y, np.abs(y), 0))
