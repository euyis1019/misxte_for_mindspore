import mindspore
import numpy as np
from mindspore import Tensor
class mft:

    def __init__(self):
        self.ops = mindspore.ops
        self.nn = mindspore.nn
    def tensor_equal(self,tensor1: Tensor, tensor2: Tensor):
        """
        Check if two tensors are equal.

        Args:
            tensor1 (mindspore.Tensor): The first tensor to compare.
            tensor2 (mindspore.Tensor): The second tensor to compare.

        Returns:
            bool: True if tensors are equal, False otherwise.
        """
        # Check if the shapes of the two tensors are equal
        if tensor1.shape != tensor2.shape:
            return False

        # Check if the data types of the two tensors are equal
        if tensor1.dtype != tensor2.dtype:
            return False

        # Check if the tensors are on the same device
        if tensor1.device != tensor2.device:
            return False

        # Element-wise comparison
        try:
            equal = self.ops.Equal()
            equal_result = equal(tensor1, tensor2)
        except Exception as e:
            print(f"Error during element-wise comparison: {str(e)}")
            return False

        # Check if all elements are True
        try:
            reduce_sum = self.ops.ReduceSum()
            sum_result = reduce_sum(equal_result, ())
        except Exception as e:
            print(f"Error during sum reduction: {str(e)}")
            return False

        return sum_result == tensor1.size


if __name__ == "__main__":
    x = Tensor(np.array([[0, 1], [2, 3], [4, 5]]))
    print(x.shape)