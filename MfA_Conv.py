import torch
import numpy as np

per_num = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
# per_num = np.array([4, 3, 2])
per_num = torch.from_numpy(per_num)
test_num = np.array([[[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]],
                     [[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]],
                     [[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]]])
test_num = torch.from_numpy(test_num)
final_num = per_num * test_num
print(final_num)


