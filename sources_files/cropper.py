import torch
import torchvision

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

#print(torch.cuda.memory_allocated())
#print(torch.cuda.empty_cache())
#print(torch.cuda.memory_allocated())

#mylist = [[3, 5, 8, 4], [6, 9, 8, 7], [2, 9, 4, 1], [6, 8, 5, 9]]
#x_first = sorted(mylist, key=lambda x: (x[0]))
#print(x_first)

#then_y = sorted(x_first, key=lambda x: (x[1]))
#print(then_y)