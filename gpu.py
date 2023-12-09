import torch

# Check if a GPU is available
cuda = torch.cuda.is_available()
print(cuda)