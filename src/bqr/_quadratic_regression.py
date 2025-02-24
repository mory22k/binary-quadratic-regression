from typing import Optional
from numpy.typing import NDArray

import numpy as np
import dimod

from .child_regression import BaseChildRegression, LstsqChildRegression


class BinaryQuadraticRegression(dimod.BinaryQuadraticModel):
    """Regression model on a complete graph represented as a Binary Quadratic Model (BQM).

    Each sample is a binary vector x in {0,1}^n and the energy is modeled as

        E(x) = offset + sum_{i <= j} Q_{ij} * (x_i * x_j),

    where the diagonal terms (i == j) are interpreted as linear biases and the off-diagonal
    terms (i > j) as quadratic interactions.

    The regression is performed by first zero-centering the target y (by subtracting its mean)
    and fitting a regression model on the transformed features (without an intercept term). The
    final intercept is then computed as the sum of y's mean and the child regressor's intercept_.
    This class inherits from dimod.BinaryQuadraticModel and its learned parameters are stored in the
    attributes `linear`, `quadratic`, and `offset`.

    Attributes:
        child_regression (BaseChildRegression): The regression model used to fit the transformed features.
        is_fitted (bool): Flag indicating whether the model has been fitted.
    """

    def __init__(self, child_regression: Optional[BaseChildRegression] = None) -> None:
        """Initializes the BinaryQuadraticRegression instance.

        Args:
            child_regression (Optional[BaseChildRegression]): An instance of a regression model implementing
                the BaseChildRegression interface. If None, a LstsqChildRegression is used.
        """
        if child_regression is None:
            child_regression = LstsqChildRegression()
        self.child_regression = child_regression
        self.is_fitted = False
        # Initialize an empty BQM with no linear or quadratic terms and zero offset.
        super().__init__({}, {}, 0.0, vartype="BINARY")

    @staticmethod
    def _transform_features(
        X: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Transforms the input binary matrix X into features for regression.

        For each sample, the features consist of all products x_i * x_j for i <= j.
        The constant bias term is NOT included here and is handled separately.

        Args:
            X (NDArray): Binary input matrix of shape (num_samples, n).

        Returns:
            tuple[NDArray, NDArray, NDArray]:
                - Transformed feature matrix of shape (num_samples, n(n+1)/2).
                - Row indices of the quadratic terms.
                - Column indices of the quadratic terms.
        """
        num_samples, d = X.shape
        # Compute the outer products for all samples: shape (num_samples, d, d)
        XX = np.einsum("bi,bj->bij", X, X)
        # Get indices for the lower triangular part (including diagonal)
        row_idx, col_idx = np.tril_indices(d)
        # Extract lower triangular elements for each sample: shape (num_samples, d(d+1)/2)
        X_quad = XX[:, row_idx, col_idx].reshape(num_samples, -1)
        return X_quad, row_idx, col_idx

    def fit(self, X: NDArray, y: NDArray) -> "BinaryQuadraticRegression":
        """Fits the regression model and constructs the BQM.

        The target vector y is zero-centered by subtracting its mean, and the child regression model
        is fit on the transformed features (without an intercept term). The final offset is computed
        as the sum of y's mean and the child regressor's intercept_.

        Args:
            X (NDArray): Binary matrix of shape (num_samples, n), where each row represents a binary assignment of vertices.
            y (NDArray): Target energy values of shape (num_samples,).

        Returns:
            BinaryQuadraticRegression: The fitted regression model.
        """
        X_trans, row_idx, col_idx = self._transform_features(X)

        self.child_regression.fit(X_trans, y)
        coef = self.child_regression.coef_
        offset = self.child_regression.intercept_

        diag_mask = row_idx == col_idx
        linear_indices = row_idx[diag_mask]
        linear_values = coef[diag_mask]
        linear = dict(zip(linear_indices, linear_values))

        quad_mask = row_idx > col_idx
        quadratic_keys = list(zip(row_idx[quad_mask], col_idx[quad_mask]))
        quadratic_values = coef[quad_mask]
        quadratic = dict(zip(quadratic_keys, quadratic_values))

        self.clear()
        self.add_linear_from(linear)
        self.add_quadratic_from(quadratic)
        self.offset += offset

        self.is_fitted = True
        return self

    def predict(self, X: NDArray) -> np.float64 | NDArray:
        """Predicts the energy of the input binary matrix.

        Args:
            X (NDArray): Binary matrix of shape (num_samples, n), where each row represents a binary assignment of vertices.

        Returns:
            np.float64 | NDArray: Predicted energy values of shape (num_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before calling predict.")

        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(1, -1)

        energies: NDArray[np.float64] = self.energies(X)

        if is_1d:
            return np.float64(energies[0])

        return energies
