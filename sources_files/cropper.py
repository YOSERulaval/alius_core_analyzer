import torch
import torchvision

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

print(torch.cuda.memory_allocated())
print(torch.cuda.empty_cache())
print(torch.cuda.memory_allocated())