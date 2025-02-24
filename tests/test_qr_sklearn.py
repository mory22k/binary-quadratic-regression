from typing import Any
from numpy.typing import NDArray
import numpy as np
from sklearn.linear_model import LinearRegression
from dimod import BinaryQuadraticModel

from bqr.child_regression import BaseChildRegression
from bqr import BinaryQuadraticRegression, bqm_to_coefs


class ChildRegression(BaseChildRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = LinearRegression()

    def fit(self, X: NDArray, y: NDArray, **kwargs: Any) -> "ChildRegression":
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.is_fitted = True
        return self


def test_qr_sklearn() -> None:
    d = 10
    n = 100
    Q_true = np.random.standard_normal((d, d))
    bqm = BinaryQuadraticModel.from_qubo(Q_true)  # type: ignore

    X_train = np.random.randint(0, 2, (n, d))
    Y_train = bqm.energies(X_train)

    bqr = BinaryQuadraticRegression(child_regression=ChildRegression())
    bqr.fit(X_train, Y_train)

    coefs_true, _, _ = bqm_to_coefs(bqm)
    coefs_pred, _, _ = bqm_to_coefs(bqr)

    assert np.allclose(coefs_true, coefs_pred), "Training failed"


if __name__ == "__main__":
    test_qr_sklearn()
