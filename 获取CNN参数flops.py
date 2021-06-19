import torch
from torchvision.models import densenet201,resnet50,resnet18
from ptflops import get_model_complexity_info

model=densenet201()
#model=resnet18()
macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))