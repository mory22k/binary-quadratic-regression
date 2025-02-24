import numpy as np
from dimod import BinaryQuadraticModel
from bqr import BinaryQuadraticRegression
from dwave.samplers import SimulatedAnnealingSampler


def test_bqr_sa() -> None:
    d = 10
    n = 100
    Q_true = np.random.standard_normal((d, d))
    bqm = BinaryQuadraticModel.from_qubo(Q_true)  # type: ignore

    X_train = np.random.randint(0, 2, (n, d))
    Y_train = bqm.energies(X_train)

    bqr = BinaryQuadraticRegression()
    bqr.fit(X_train, Y_train)

    sa_sampler = SimulatedAnnealingSampler()
    sampleset_pred = sa_sampler.sample(bqr, num_reads=10)
    sampleset_true = sa_sampler.sample(bqm, num_reads=10)

    print(sampleset_pred)
    print(sampleset_true)


if __name__ == "__main__":
    test_bqr_sa()
