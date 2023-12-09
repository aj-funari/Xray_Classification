import torch

# Check if a GPU is available
def check_gpu_availability():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    print(check_gpu_availability())