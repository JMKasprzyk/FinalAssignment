import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class conv_block_2x2(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block_2x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=2, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x

# 3x3 and 7x7 convolutional layers blocks as in paper
    
class conv_block_3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block_3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class conv_block_5x5(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block_5x5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class conv_block_7x7(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block_7x7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x
    

class conv_block_9x9(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block_9x9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=9, padding=4, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=9, padding=4, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x
    
class conv_block_3x3_dilated(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block_3x3_dilated, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x
    
class MS_block_37(nn.Module):
    def __init__(self, in_c, out_c):
        super(MS_block_37, self).__init__()
        
        self.conv_block_3x3 = conv_block_3x3(in_c, out_c)
        self.conv_block_7x7 = conv_block_7x7(in_c, out_c)

        self.conv_1x1 = nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inputs):
        x3 = self.conv_block_3x3(inputs)
        x7 = self.conv_block_7x7(inputs)
        x = torch.cat([x3, x7], axis=1)
        x = self.conv_1x1(x)
        return x
    
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
            self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, inputs):
        x = self.up(inputs)

        return x
    

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = MS_block_37(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = deconv_block_2x2(in_c, out_c)
        self.conv = MS_block_37(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class MSU_Net(nn.Module):
    def __init__(self, in_c=3, out_c=64):
        super(MSU_Net, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        """Encoder"""
        self.enc1 = encoder_block(in_c, filters[0])
        self.enc2 = encoder_block(filters[0], filters[1])
        self.enc3 = encoder_block(filters[1], filters[2])
        self.enc4 = encoder_block(filters[2], filters[3])

        """Bottleneck"""
        self.bottle = MS_block_37(filters[3], filters[4])
        
        """Decoder"""
        self.dec1 = decoder_block(filters[4], filters[3])
        self.dec2 = decoder_block(filters[3], filters[2])
        self.dec3 = decoder_block(filters[2], filters[1])
        self.dec4 = decoder_block(filters[1], filters[0])
        
        """Classifier"""
        self.out = nn.Conv2d(filters[0], out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.enc1(inputs)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        """ Bottleneck """
        b = self.bottle(p4)

        """ Decoder """
        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        """ Segmentation output """
        outputs = self.out(d4)

        return outputs