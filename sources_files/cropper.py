import torch
import torchvision

print(torch.__version__)
print(torch.version.cuda) # Regarde si CUDA est bien installé
print(torch.cuda.is_available())
torch.cuda.empty_cache()