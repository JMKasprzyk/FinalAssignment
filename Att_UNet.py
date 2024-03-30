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

class attention_gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_gate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Attention_Block(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(Attention_Block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels[0], inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels[1], inter_channels, kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # Vector 'g' goes through a 1x1x1 convolution to retain its dimensions
        Wg = self.W_g(g)

        # Vector 'x' goes through a stided convolution to reduce its dimensions
        Wx = self.W_x(x)

        # Element-wise addition of the gating and x signals
        add_xg = self.relu(Wg + Wx)

        # 1x1x1 convolution + Sigmoid activation
        psi = self.psi(add_xg)

        return psi * x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att = Attention_Block(in_c, out_c)
        self.conv = conv_block(in_c[0] + out_c, out_c)

    def forward(self, inputs, skip):
        print("Shape of skip: ", skip.shape)
        x = self.up(inputs)
        print("Shape of upsampled input: ", x.shape)
        att = self.Att(x, skip)
        print("Shape of attention output: ", att.shape)
        x = torch.cat([x, att], axis=1)
        x = self.conv(x)

        return x
    
class Att_UNet(nn.Module):
    def __init__(self, in_c=3, out_c=34):
        super(Att_UNet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.encoder1 = encoder_block(in_c, filters[0])
        self.encoder2 = encoder_block(filters[0], filters[1])
        self.encoder3 = encoder_block(filters[1], filters[2])
        self.encoder4 = encoder_block(filters[2], filters[3])

        self.bottleneck = conv_block(filters[3], filters[4])

        self.decoder1 = decoder_block([filters[4], filters[3]], filters[3])
        self.decoder2 = decoder_block([filters[3], filters[2]], filters[2])
        self.decoder3 = decoder_block([filters[2], filters[1]], filters[1])
        self.decoder4 = decoder_block([filters[1], filters[0]], filters[0])

        self.conv = nn.Conv2d(filters[0], out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.encoder1(inputs)
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
    x = torch.randn((8, 3, 512, 512)) # Batch size of 8, 3 channels, 512x512 image
    model = Att_UNet()
    y = model(x)
    print(y.size())
