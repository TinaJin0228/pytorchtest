import torch
import torchvision
import numpy as np

a = torch.Tensor([1.])
# tensor initialization
# directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\n{x_ones}\n")

x_rand = torch.rand_like(x_data,dtype = torch.float)
print(f"Random Tensor: \n{x_rand}\n")

# with random or constant values:
# shape is the tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

# tensor attributes: shape, device, datatype
tensor = torch.rand(2,3)
print(f"shape of tensor:{tensor.shape}")
print(f"datatype of tensor:{tensor.dtype}")
print(f"device tensor is stored on:{tensor.device}")

