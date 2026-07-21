import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = np.max(z, axis = 0)
        sum_z = np.sum(np.exp(z - max_z))
        return np.round((np.exp(z - max_z))/(sum_z), 4)

