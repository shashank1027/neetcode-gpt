import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        loss=-((y_true)*np.log(y_pred+epsilon)+(1-y_true)*np.log(1-y_pred+epsilon))
        return round(np.mean(loss),4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        loss = -np.sum(
    y_true * np.log(y_pred + epsilon),
    axis=1
)
        return round(np.mean(loss),4)
        pass
