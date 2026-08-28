import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z=np.dot(x,w)+b
        y_pred=1/(1+np.exp(-z))
        error=y_pred-y_true
        sigmoid_grad = y_pred * (1 - y_pred)
        dL_dz = error * sigmoid_grad
        dL_dw = dL_dz * x
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)
        

        pass
