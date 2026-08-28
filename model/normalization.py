import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        mean = np.mean(x)
        centered = x - mean
        variance = np.mean(centered ** 2)
        std = np.sqrt(variance + 1e-5)
        normalized = centered / std
        output = normalized * gamma + beta
        return np.round(output, 5)

        
