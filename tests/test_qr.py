import numpy as np
from dimod import BinaryQuadraticModel
from bqr import BinaryQuadraticRegression, bqm_to_coefs
from bqr.child_regression import NaiveLstsqChildRegression


def test_bqr() -> None:
    d = 10
    n = 200
    Q_true = np.random.standard_normal((d, d))
    bqm = BinaryQuadraticModel.from_qubo(Q_true)  # type: ignore

    X_train = np.random.randint(0, 2, (n, d))
    Y_train = bqm.energies(X_train)

    bqr = BinaryQuadraticRegression()
    bqr_naive = BinaryQuadraticRegression(child_regression=NaiveLstsqChildRegression())
    bqr.fit(X_train, Y_train)
    bqr_naive.fit(X_train, Y_train)

    coefs_true, _, _ = bqm_to_coefs(bqm)
    coefs_pred, _, _ = bqm_to_coefs(bqr)
    coefs_naive, _, _ = bqm_to_coefs(bqr_naive)

    assert np.allclose(coefs_true, coefs_pred), "Training failed"
    assert np.allclose(coefs_true, coefs_naive), "Training failed"

    print(Y_train - bqr.predict(X_train))

if __name__ == "__main__":
    test_bqr()
