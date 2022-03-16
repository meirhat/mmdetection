import torch
from torchvision import utils
import matplotlib.pyplot as plt


grid = utils.make_grid(model.backbone.conv1.weight, nrow=8, normalize=True, padding=2)

plt.figure()
plt.imshow(grid.numpy().transpose((1, 2, 0)))

model.backbone.conv1.weight

for module in model.backbone.modules():
    if isinstance(module, torch.nn.Conv2d):
        print(module)


for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        print(module.weight)


for module in model.backbone.modules():
        print(module)


for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        print(module.state_dict())


# show a filter in a specific layer:
j=0
for i, module in enumerate(model.modules()):
    if isinstance(module, torch.nn.Conv2d):
        j = j + 1
        if j==6:
            grid = utils.make_grid(module.weight, nrow=8, normalize=True, padding=2)
            print('kernel size: ', module.kernel_size)
            print('input channels: ', module.in_channels)
            print('out channels: ', module.out_channels)
            print('grid shape: ', grid.shape)
            plt.figure()
            # plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.imshow(grid[0,:,:].numpy())
        # print(j, module)