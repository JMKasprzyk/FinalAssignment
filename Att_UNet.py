import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_rate=None):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity(),
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x
    

class AttentionGate(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Attention Gate for Attention UNet, which allows the network to focus on the certatin regions of the input image during training.
        Implemenation based on the paper: https://arxiv.org/pdf/1804.03999.pdf 

        Args:
            in_ch (list):   A list containing two elements. The first element is the number of channels in 'g' 
                            (the output of the previous layer), and the second element is the number of channels 
                            in 'x' (the skip connection from the encoder).
            out_ch (int):   The number of output channels for the attention gate.
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_ch[0], out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch[1], out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, inputs):
        Wg = self.W_g(g)
        Wx = self.W_x(inputs)

        psi = self.relu(Wg + Wx)
        psi = self.psi(psi)

        return inputs * psi

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, drop_out=None):
        super().__init__()

        self.conv = conv_block(in_c, out_c, dropout_rate=drop_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att = AttentionGate(in_c, out_c)
        self.conv = conv_block(in_c[0] + out_c, out_c)

    def forward(self, inputs, skip):
        # print("Shape of skip: ", skip.shape)
        x = self.up(inputs)
        # print("Shape of upsampled input: ", x.shape)
        att = self.Att(x, skip)
        # print("Shape of attention output: ", att.shape)
        x = torch.cat([x, att], axis=1)
        x = self.conv(x)

        return x
    
class Att_UNet(nn.Module):
    def __init__(self, in_c=3, out_c=19):
        super(Att_UNet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.encoder1 = encoder_block(in_c, filters[0], drop_out=0.3)
        self.encoder2 = encoder_block(filters[0], filters[1], drop_out=0.1)
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
    x = torch.randn((4, 3, 128, 128)) # Batch size of 8, 3 channels, 512x512 image
    model = Att_UNet()
    y = model(x)
    print(y.size())
