import numpy as np
from numpy.typing import NDArray
from dimod import BinaryQuadraticModel


def bqm_to_coefs(
    bqm: BinaryQuadraticModel,
) -> tuple[NDArray, NDArray, NDArray]:
    linear_biases, quadratic, offset = bqm.to_numpy_vectors(return_labels=False)  # type: ignore
    (row_idx, col_idx, quadratic_biases) = quadratic
    coefs = np.concatenate([[offset], linear_biases, quadratic_biases])
    return coefs, row_idx, col_idx
