import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        sigmoid= 1/(1+np.exp(-z))
        return np.round(sigmoid, 5)
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        ReLU=np.maximum(0,z)
        return np.round(ReLU,5)
        pass
