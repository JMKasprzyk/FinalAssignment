import torch
import torch.nn as nn


class deconv_block_2x2(nn.Module):
    def __init__(self, in_c, out_c, bilinear=False):
        super(deconv_block_2x2, self).__init__()
        if bilinear:
            self.deconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, inputs):
        x = self.deconv(inputs)

        return x


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
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


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

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
    def __init__(self, in_c, out_c, t):
        super().__init__()

        self.conv = RRCNN_block(in_c, out_c, t)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, t):
        super().__init__()

        self.up = deconv_block_2x2(in_c, out_c, bilinear=True)
        self.conv = RRCNN_block(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
class R2U_Net(nn.Module):
    def __init__(self,in_c=3,out_c=34,t=2):
        super(R2U_Net,self).__init__()

        filters = [64, 128, 256, 512, 1024]

        """Encoder"""
        self.encoder1 = encoder_block(in_c, filters[0], t)
        self.encoder2 = encoder_block(filters[0], filters[1], t)
        self.encoder3 = encoder_block(filters[1], filters[2], t)
        self.encoder4 = encoder_block(filters[2], filters[3], t)
        """Code"""
        self.bottleneck = RRCNN_block(filters[3], filters[4], t)
        "Decoder"
        self.decoder4 = decoder_block(filters[4], filters[3], t)
        self.decoder3 = decoder_block(filters[3], filters[2], t)
        self.decoder2 = decoder_block(filters[2], filters[1], t)
        self.decoder1 = decoder_block(filters[1], filters[0], t)
        """Classifier"""
        self.conv_1x1 = nn.Conv2d(filters[0], out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.encoder1(inputs)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        print("Passed the Encoder")

        """ Bottleneck """
        b = self.bottleneck(p4)

        print("Passed the Bottleneck")

        """ Decoder """
        d1 = self.decoder4(b, s4)
        d2 = self.decoder3(d1, s3)
        d3 = self.decoder2(d2, s2)
        d4 = self.decoder1(d3, s1)

        print("Passed the Decoder")

        """ Segmentation output """
        outputs = self.conv_1x1(d4)

        print("Passed the Classifier")

        return outputs