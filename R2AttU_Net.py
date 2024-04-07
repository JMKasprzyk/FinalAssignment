import torch.nn as nn
import torch

class conv_block(nn.Module):
    """
    Convolution Block 
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
    Up Convolution Block
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
    Recurrent Block
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
    Recurrent Residual Convolutional Neural Network Block
    Implementated based on the paper: https://arxiv.org/abs/1802.06955
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

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class encoder_block(nn.Module):
    """
    Encoder Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(encoder_block, self).__init__()

        self.R2_conv = Recurrent_Residual_block(in_ch, out_ch, t=t)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.R2_conv(x)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    """
    Decoder Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(decoder_block, self).__init__()

        self.up = deconv_block(in_ch, out_ch)
        self.att = Attention_block(F_g=out_ch, F_l=out_ch, F_int=out_ch//2) # the # of features in the intermediate layer is half of the input feature map
        self.R2_conv = Recurrent_Residual_block(in_ch, out_ch, t=t)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(g=x, x=skip)
        x = torch.cat([x, skip], axis=1)
        x = self.R2_conv(x)
        return x
    
class R2AttU_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=34, t=2):
        super(R2AttU_Net, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.encoder1 = encoder_block(in_ch, filters[0], t)
        self.encoder2 = encoder_block(filters[0], filters[1], t)
        self.encoder3 = encoder_block(filters[1], filters[2], t)
        self.encoder4 = encoder_block(filters[2], filters[3], t)

        self.bottleneck = Recurrent_Residual_block(filters[3], filters[4], t)

        self.decoder1 = decoder_block(filters[4], filters[3], t)
        self.decoder2 = decoder_block(filters[3], filters[2], t)
        self.decoder3 = decoder_block(filters[2], filters[1], t)
        self.decoder4 = decoder_block(filters[1], filters[0], t)

        self.conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        bottle = self.bottleneck(p4)

        d1 = self.decoder1(bottle, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        out = self.conv(d4)

        return out

if __name__ == '__main__':
    x = torch.randn((4, 3, 128, 128)) # Batch size of 4, 3 channels [RGB], 128x128 image
    model = R2AttU_Net()
    y = model(x)
    print(y.size())
