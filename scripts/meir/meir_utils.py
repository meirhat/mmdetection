import torch
from torchvision import utils
import matplotlib.pyplot as plt

def show_filter(model, layer_num, channel_num):
    j = 0
    for i, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.Conv2d):
            j = j + 1
            if j == 6:
                grid = utils.make_grid(module.weight, nrow=8, normalize=True, padding=2)
                print('kernel size: ', module.kernel_size)
                print('input channels: ', module.in_channels)
                print('out channels: ', module.out_channels)
                print('grid shape: ', grid.shape)
                plt.figure()
                # plt.imshow(grid.numpy().transpose((1, 2, 0)))
                plt.imshow(grid[0, :, :].numpy())


# if __name__ == 'main':
#     show_filter(model, layer_num, channel_num)