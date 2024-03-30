import torch
import torch.nn as nn
import math

""""
Implemented in this scripts Deep ResUnet Architecture is from the paper: https://arxiv.org/pdf/1711.10684.pdf
Fig 1. -> Building blocks of the proposed Deep ResUNet architecture
Fig 2. + TABLE 1 -> Deep ResUNet structure
"""

# # def same_padding(input_size, kernel_size, stride):
# #     output_size = math.ceil(float(input_size) / float(stride))
# #     padding = max(0, (output_size - 1) * stride + kernel_size - input_size)
# #     return padding // 2

# def same_padding(kernel_size, stride):
#     return (kernel_size - 1) // 2 if stride == 1 else 0

class BN_ReLU(nn.Module):
    """
    BarchNorm -> ReLU activation Block with optional activation
    """
    def __init__(self, in_ch, activation=True):
        super(BN_ReLU, self).__init__()
        self.act_state = activation
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        if self.act_state:
            out = self.relu(out)
        return out
    
class Conv_block(nn.Module):
    """
    Convolutional Block with BatchNorm and ReLU activation
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(Conv_block, self).__init__()
        self.bn_relu = BN_ReLU(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=1)

    def forward(self, x):
        x = self.bn_relu(x)
        x = self.conv(x)
        return x
    
class Stem_block(nn.Module):
    """
    Stem Block for Deep ResUNet
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, strides=1):
        super(Stem_block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, strides, padding=1)
        self.CB = Conv_block(out_ch, out_ch, kernel_size, strides)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=strides, padding=0)
        self.bn = BN_ReLU(out_ch, activation=False)
        
    def forward(self, x):
        print("Stem input shape: ", x.shape)
        c1 = self.conv(x)
        CB_out = self.CB(c1)
        print("Conv Block output shape: ", CB_out.shape)

        shortcut = self.shortcut(x)
        shortcut = self.bn(shortcut)
        print("Shortcut shape: ", shortcut.shape)

        # Addition
        output = CB_out + shortcut

        return output


class Residual_block(nn.Module):
    """
    Residual Block for Deep ResUNet
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, strides=1):
        super(Residual_block, self).__init__()
        self.CB1 = Conv_block(in_ch, out_ch, kernel_size, strides)
        self.CB2 = Conv_block(out_ch, out_ch, kernel_size, stride=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=strides, padding=0)
        self.bn = BN_ReLU(out_ch, activation=False)
        
    def forward(self, x):
        CB1_out = self.CB1(x)
        CB2_out = self.CB2(CB1_out)
        print("Residual Block output shape: ", CB2_out.shape)

        shortcut = self.shortcut(x)
        shortcut = self.bn(shortcut)
        print("Shortcut shape: ", shortcut.shape)

        # Addition
        output = CB2_out + shortcut

        return output

class Upsample_Concat_block(nn.Module):
    """
    Upsample and Concatenation Block for Deep ResUNet
    """
    def __init__(self):
        super(Upsample_Concat_block, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, xskip):
        print("Upsample input shape: ", x.shape)
        u = self.up(x)
        print("Upsample output shape: ", u.shape)
        print("Skip shape: ", xskip.shape)
        c = torch.cat([u, xskip], axis=1)
        return c
    
class ResUNet(nn.Module):
    """
    Deep ResUNet Architecture
    """
    def __init__(self, in_ch=3, out_ch=34):
        super(ResUNet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        """ Encoder """
        self.stem = Stem_block(in_ch, filters[0])
        self.res1 = Residual_block(filters[0], filters[1], strides=2)
        self.res2 = Residual_block(filters[1], filters[2], strides=2)
        self.res3 = Residual_block(filters[2], filters[3], strides=2)
        self.res4 = Residual_block(filters[3], filters[4], strides=2)

        """ Bridge """
        self.bridge = nn.Sequential(
            Conv_block(filters[4], filters[4],stride=1),
            Conv_block(filters[4], filters[4],stride=1)
        )

        """ Decoder """
        self.up_concat4 = Upsample_Concat_block()
        self.res5 = Residual_block(filters[4], filters[3])
        self.up_concat3 = Upsample_Concat_block()
        self.res6 = Residual_block(filters[3], filters[2])
        self.up_concat2 = Upsample_Concat_block()
        self.res7 = Residual_block(filters[2], filters[1])
        self.up_concat1 = Upsample_Concat_block()
        self.res8 = Residual_block(filters[1], filters[0])

        """ Classifier """
        self.out_conv = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):

        """ Encoder """
        e0 = self.stem(x)
        print("PASSED STEM ")
        e1 = self.res1(e0)
        print("PASSED ENCODER 1")
        e2 = self.res2(e1)
        print("PASSED ENCODER 2")
        e3 = self.res3(e2)
        print("PASSED ENCODER 3")
        e4 = self.res4(e3)
        print("PASSED ENCODER 4")
        
        """ Bridge """
        b = self.bridge(e4)
        print("PASSED BRIDGE")

        """ Decoder """
        d1 = self.up_concat4(b, e4)
        print("PASSED UP CONCAT 4 : ", d1.shape)
        d1 = self.res5(d1)
        print("PASSED RES 5: ", d1.shape)
        d2 = self.up_concat3(d1, e3)
        print("PASSED UP CONCAT 3: ", d2.shape)
        d2 = self.res6(d2)
        print("PASSED RES 6: ", d2.shape)
        d3 = self.up_concat2(d2, e2)
        print("PASSED UP CONCAT 2: ", d3.shape)
        d3 = self.res7(d3)
        print("PASSED RES 7: ", d3.shape)
        d4 = self.up_concat1(d3, e1)
        print("PASSED UP CONCAT 1: ", d4.shape)
        d4 = self.res8(d4)
        print("PASSED RES 8: ", d4.shape)

        """ Segmentaiton output """
        out = torch.sigmoid(self.out_conv(d4))
        return out
    
if __name__ == '__main__':
    x = torch.randn((2, 3, 128, 128)) # Batch size of 8, 3 channels, 512x512 image
    model = ResUNet()
    y = model(x)
    print(y.size())

class Upsample_Concat_block(nn.Module):
    """
    Upsample and Concatenation Block for Deep ResUNet
    """
    def __init__(self, input, skip):
        super(Upsample_Concat_block, self).__init__()
        self.input = input
        self.skip = skip
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self):
        print("Upsample input shape: ", self.input.shape)
        u = self.up(self.input)
        print("Upsample output shape: ", u.shape)
        print("Skip shape: ", self.skip.shape)
        c = torch.cat([u, self.skip], axis=1)
        return c