import torch
import torch.nn as nn

class Res_block(nn.Module):
    """
    Residual Block for R2Unet_CNN 
    checkout: Fig 4.: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf

    Conv_Layer -> BN -> ReLU -> Conv_Layer -> BN -> shortcut -> BN -> shortcut+BN -> ReLU
    """
    def __init__(self, in_ch, out_ch, filter_size):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=filter_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=filter_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        
        self.shorcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(out_ch))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shorcut(residual)

        out += shortcut
        out = self.relu(out)

        return out