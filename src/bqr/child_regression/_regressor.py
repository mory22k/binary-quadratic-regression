from typing import Any
import abc

import numpy as np
from numpy.typing import NDArray


class BaseChildRegression(abc.ABC):
    """Abstract base class for regression models.

    Attributes:
        coef_ (NDArray): Coefficient vector of shape (num_features,).
        intercept_ (float): Intercept term.
        is_fitted (bool): Flag indicating whether the model has been fitted.

    Methods:
        fit(X: NDArray, y: NDArray) -> "BaseChildRegression":
            Fit the regression model using the feature matrix X and target vector y.
        predict(X: NDArray) -> NDArray:
            Predict target values for the given feature matrix X.
    """

    coef_: NDArray
    intercept_: float
    is_fitted: bool

    @abc.abstractmethod
    def fit(self, X: NDArray, y: NDArray, **kwargs: Any) -> "BaseChildRegression":
        """Fits the model to the data.

        Args:
            X (NDArray): Feature matrix of shape (num_samples, num_features).
            y (NDArray): Target vector of shape (num_samples,).

        Returns:
            BaseChildRegression: The fitted regression model.
        """
        pass


class LstsqChildRegression(BaseChildRegression):
    """A simple linear regressor using the least squares solution computed via np.linalg.lstsq.

    Attributes:
        coef_ (NDArray): Coefficient vector of shape (num_features,).
        intercept_ (float): Intercept term, set to zero since the model does not include an intercept column.
        is_fitted (bool): Flag indicating whether the model has been fitted.
    """

    def __init__(self) -> None:
        self.coef_ = np.array([])
        self.intercept_ = 0.0
        self.is_fitted = False

    def fit(self, X: NDArray, y: NDArray, **kwargs: Any) -> "LstsqChildRegression":
        """Fits the linear model using the least squares solution.

        The function centers X and y by subtracting their respective means, solves the least squares problem
        on the centered data, and then computes the intercept from the means.

        Args:
            X (NDArray): Feature matrix of shape (num_samples, num_features).
            y (NDArray): Target vector of shape (num_samples,).

        Returns:
            LstsqChildRegression: The fitted regressor instance.
        """
        X_mean: NDArray = np.mean(X, axis=0)
        y_mean = np.mean(y)

        X_centered = X - X_mean
        y_centered = y - y_mean

        solution, _, _, _ = np.linalg.lstsq(X_centered, y_centered, rcond=None)

        self.coef_ = solution
        self.intercept_ = y_mean - X_mean @ solution

        self.is_fitted = True
        return self

class NaiveLstsqChildRegression(BaseChildRegression):
    def fit(self, X: NDArray, y: NDArray, **kwargs: Any) -> "NaiveLstsqChildRegression":
        X_extended = np.hstack((X, np.ones((X.shape[0], 1))))
        solution, _, _, _ = np.linalg.lstsq(X_extended, y, rcond=None)

        self.coef_ = solution[:-1]
        self.intercept_ = solution[-1]

        self.is_fitted = True
        return self
