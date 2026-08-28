import torch
import torch.nn
from torchtyping import TensorType
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M, N = to_reshape.shape
        rows=(M*N//2)
        columns=2
        shae=rows,columns
        new_shape=torch.reshape(to_reshape, shae)
        return new_shape
    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        avg=torch.mean(to_avg,dim=0)
        return avg
        pass
    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        concat=torch.cat((cat_one, cat_two), dim=1)
        return concat
        pass
    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        loss=torch.nn.functional.mse_loss(prediction, target)
        return torch.round(loss,decimals=4)
        pass
