import torch.nn as nn
import torch

class conv_block(nn.Module):
    """
    Convolution Block - This block applies two convolution operations on the input tensor,
    each followed be a batch normalization and ReLU activation function.

    conv_block == 2D Conv_Layer -> BatchNorm -> ReLU -> 2D Conv_Layer -> BatchNorm -> ReLU

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class deconv_block(nn.Module):
    """
    Up Convolution Block - This block upsamples the input tensor and applies a convolution operation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """
    def __init__(self, in_ch, out_ch):
        super(deconv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN: 
    This block applies a convolution operation on the input tensor for a specified number of times.

    Args:
        out_ch (int): Number of output channels.
        t (int): Number of times the operation is repeated.
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class Recurrent_Residual_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block:
    This block applies a recurrent block operation on the input tensor and adds the result to the original input.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        t (int): Number of times the operation is repeated.
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(Recurrent_Residual_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out
    
class encoder_block(nn.Module):
    """
    Encoder Block:
    This block applies a recurrent residual block operation and a max pooling operation on the input tensor.
    Retunrs the output of the recurrent residual block (to be used in a skip connection by the decoder) and the max pooling operation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        t (int): Number of times the operation is repeated.
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(encoder_block, self).__init__()

        self.R2_conv = Recurrent_Residual_block(in_ch, out_ch, t=t)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.R2_conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    """
    Decoder Block: 
    This block upsamples the input tensor, concatenates it with a skip connection, and applies
    a recurrent residual block operation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        t (int): Number of times the operation is repeated.
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(decoder_block, self).__init__()

        self.up = deconv_block(in_ch, out_ch)
        self.R2_conv = Recurrent_Residual_block(in_ch, out_ch, t=t)

    def forward(self, input, skip):
        x = self.up(input)
        x = torch.cat([x, skip], axis=1)
        x = self.R2_conv(x)
        return x

class R2U_Net(nn.Module):
    """
    R2U-Unet Structure definition:
    This class defines the structure of the R2U-Unet model.

    Args:
        img_ch (int): Number of input image channels.
        output_ch (int): Number of output channels.
        t (int): Number of times the operation is repeated in recurrent blocks.

    Implemented based o nthe paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=3, output_ch=34, t=2):
        super(R2U_Net, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.encoder1 = encoder_block(img_ch, filters[0], t=t)
        self.encoder2 = encoder_block(filters[0], filters[1], t=t)
        self.encoder3 = encoder_block(filters[1], filters[2], t=t)
        self.encoder4 = encoder_block(filters[2], filters[3], t=t)

        self.bottleneck = Recurrent_Residual_block(filters[3], filters[4], t=t)

        self.decoder4 = decoder_block(filters[4], filters[3], t=t)
        self.decoder3 = decoder_block(filters[3], filters[2], t=t)
        self.decoder2 = decoder_block(filters[2], filters[1], t=t)
        self.decoder1 = decoder_block(filters[1], filters[0], t=t)

        self.final_conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.encoder1(inputs)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b = self.bottleneck(p4)

        d1 = self.decoder4(b, s4)
        d2 = self.decoder3(d1, s3)
        d3 = self.decoder2(d2, s2)
        d4 = self.decoder1(d3, s1)

        out = self.final_conv(d4)

        return out

if __name__ == '__main__':
    x = torch.randn((4, 3, 128, 128)) # Batch size of 4, 3 channels [RGB], 128x128 image
    model = R2U_Net()
    y = model(x)
    print(y.size())
