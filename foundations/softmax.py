import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z=z-np.max(z)
        exp_values=np.exp(z)
        total=np.sum(exp_values)
        softmax=exp_values/total
        return np.round(softmax,4)
        pass
