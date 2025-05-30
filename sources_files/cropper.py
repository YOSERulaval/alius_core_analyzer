import torch
import torchvision

print(torch.__version__)        # ex: '2.1.0'
print(torch.version.cuda)       # ex: '11.8'

print(torchvision.ops.nms)  # doit afficher un wrapper et **pas** une erreur