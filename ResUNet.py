import torch
import torch.nn as nn

class BN_ReLU(nn.Module):
    """
    BarchNorm -> ReLU activation Block with optional activation
    """
    def __init__(self, in_ch, activation=True):
        super(BN_ReLU, self).__init__()
        self.act_state = activation
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        if self.act_state:
            x = self.relu(x)
        return x
    
class Conv_block(nn.Module):
    """
    Convolutional Block with BatchNorm and ReLU activation
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv_block, self).__init__()
        self.bn_relu = BN_ReLU(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        x = self.bn_relu(x)
        x = self.conv(x)
        return x
    
class Stem_block(nn.Module):
    """
    Stem Block for Deep ResUNet
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(Stem_block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.CB = Conv_block(out_ch, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=stride)
        self.bn = BN_ReLU(out_ch, activation=False)
        
    def forward(self, x):
        c1 = self.conv(x)
        CB_out = self.CB(c1)

        shortcut = self.shortcut(x)
        shortcut = self.bn(shortcut)

        # Addition
        output = CB_out + shortcut

        return output

class Residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.CB1 = Conv_block(in_ch, out_ch, stride=stride) # Conv_block with specified stride
        self.CB2 = Conv_block(out_ch, out_ch) # default Conv_block wirth stride=1

        """ Shortcut Connection (Identity Mapping) """
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=stride)
        """ BatchNorm """
        self.bn = BN_ReLU(out_ch, activation=False)

    def forward(self, inputs):
        CB1_out = self.CB1(inputs)
        CB2_out = self.CB2(CB1_out)

        s = self.shortcut(inputs)
        s = self.bn(s)

        # Addition
        skip = CB2_out + s
        return skip

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = Residual_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        u = self.upsample(inputs)
        c = torch.cat([u, skip], axis=1)
        output = self.r(c)
        return output
    
class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=34):
        super().__init__()

        filters = [64, 128, 256, 512]

        """ Encoder 1 """
        self.stem = Stem_block(in_ch, filters[0])

        """ Encoder 2 and 3 """
        self.r2 = Residual_block(filters[0], filters[1], stride=2)
        self.r3 = Residual_block(filters[1], filters[2], stride=2)

        """ Bridge """
        self.r4 = Residual_block(filters[2], filters[3], stride=2)

        """ Decoder """
        self.d1 = decoder_block(filters[3], filters[2])
        self.d2 = decoder_block(filters[2], filters[1])
        self.d3 = decoder_block(filters[1], filters[0])

        """ Output """
        self.output = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """ Encoder 1 """
        skip1 = self.stem(inputs)

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)
        print("Bridge shape: ", b.shape)

        """ Decoders """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ Semantic Segmentation Output """
        output = self.output(d3)
        output = self.sigmoid(output)

        return output

if __name__ == "__main__":
    inputs = torch.randn((4, 3, 512, 512))
    model = ResUNet()
    y = model(inputs)
    print(y.shape)