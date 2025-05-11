import torch
print(torch.cuda.is_available())  # True if CUDA is available, otherwise False
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the GPU (if available)
