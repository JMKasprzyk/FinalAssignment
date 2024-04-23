import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se

        expansion_channels = in_channels * expansion_rate
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            print("Error: unsupported kernel size")

        # Expansion
        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU()
            )

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm2d(expansion_channels),
            nn.ReLU()
        )

        # Squeeze and excitation block
        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=expansion_channels, out_channels=se_channels, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=se_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):

        x = inputs

        if self.expansion_rate != 1:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)

        if self.se:
            x = self.se_block(x) * x

        x = self.pointwise_conv(x)

        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs

        return x


class EffUNet(nn.Module):
    """ U-Net with EfficientNet-B0 encoder """

    def __init__(self, in_channels, classes):
        super().__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, stride=1, expansion_rate=1),
            MBConvBlock(16, 24, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock(24, 40, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock(40, 80, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 112, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 192, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 320, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.up_block_4 = DecoderBlock(432, 256)

        self.up_block_3 = DecoderBlock(296, 128)

        self.up_block_2 = DecoderBlock(152, 64)

        self.up_block_1a = DecoderBlock(96, 32)

        self.up_block_1b = DecoderBlock(32, 16)

        self.head_conv = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.start_conv(x)

        x2 = self.down_block_2(x1)

        x3 = self.down_block_3(x2)

        x4 = self.down_block_4(x3)

        x5 = self.down_block_5(x4)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x4], dim=1)

        x5 = self.up_block_4(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x3], dim=1)

        x5 = self.up_block_3(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x2], dim=1)

        x5 = self.up_block_2(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x1], dim=1)

        x5 = self.up_block_1a(x5)
        x5 = F.interpolate(x5, scale_factor=2)

        x5 = self.up_block_1b(x5)
        output = self.head_conv(x5)

        return output
# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, inputs):
#         x = self.conv(inputs)
#         return x
    

# class AttentionGate(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         """
#         Attention Gate for Attention UNet, which allows the network to focus on the certatin regions of the input image during training.
#         Implemenation based on the paper: https://arxiv.org/pdf/1804.03999.pdf 

#         Args:
#             in_ch (list):   A list containing two elements. The first element is the number of channels in 'g' 
#                             (the output of the previous layer), and the second element is the number of channels 
#                             in 'x' (the skip connection from the encoder).
#             out_ch (int):   The number of output channels for the attention gate.
#         """
#         super(AttentionGate, self).__init__()

#         self.W_g = nn.Sequential(
#             nn.Conv2d(in_ch[0], out_ch, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(out_ch)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(in_ch[1], out_ch, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(out_ch)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU()

#     def forward(self, g, inputs):
#         Wg = self.W_g(g)
#         Wx = self.W_x(inputs)

#         psi = self.relu(Wg + Wx)
#         psi = self.psi(psi)

#         return inputs * psi


    
# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.Att = AttentionGate(in_c, out_c)
#         self.conv = conv_block(in_c[0] + out_c, out_c)

#     def forward(self, inputs, skip):
#         # print("Shape of skip: ", skip.shape)
#         x = self.up(inputs)
#         # print("Shape of upsampled input: ", x.shape)
#         att = self.Att(x, skip)
#         # print("Shape of attention output: ", att.shape)
#         x = torch.cat([x, att], axis=1)
#         x = self.conv(x)

#         return x

# class Eff_Att_UNet(nn.Module):
#     def __init__(self, out_c=34):
#         super(Eff_Att_UNet, self).__init__()

#         base_model = EfficientNet.from_pretrained('efficientnet-b0')

#         self.encoder1 = base_model._conv_stem
#         self.encoder2 = base_model._blocks[0]
#         self.encoder3 = base_model._blocks[1:3]
#         self.encoder4 = base_model._blocks[3:5]

#         self.bottleneck = base_model._blocks[5:]

#         filters = [16, 24, 40, 80, 112, 192, 320]

#         self.decoder1 = decoder_block([filters[6], filters[5]], filters[5])
#         self.decoder2 = decoder_block([filters[5], filters[4]], filters[4])
#         self.decoder3 = decoder_block([filters[4], filters[3]], filters[3])
#         self.decoder4 = decoder_block([filters[3], filters[2]], filters[2])

#         self.conv = nn.Conv2d(filters[2], out_c, kernel_size=1, stride=1, padding=0)

#     def forward(self, inputs):
#         s1 = self.encoder1(inputs)
#         s2 = self.encoder2(s1)
#         s3 = self.encoder3(s2)
#         s4 = self.encoder4(s3)

#         bottle = self.bottleneck(s4)

#         d1 = self.decoder1(bottle, s4)
#         d2 = self.decoder2(d1, s3)
#         d3 = self.decoder3(d2, s2)
#         d4 = self.decoder4(d3, s1)

#         out = self.conv(d4)

#         return out
    
# if __name__ == "__main__":
#     inputs = torch.randn((4, 3, 128, 128))
#     model = Eff_Att_UNet()
#     y = model(inputs)
#     print(y.shape)