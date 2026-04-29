import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base);     self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2);    self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4);  self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8);  self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2); self.conv4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2);  self.conv3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2);  self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2);    self.conv1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        c4 = self.down4(p3); p4 = self.pool4(c4)
        bn = self.bottleneck(p4)
        u4 = self.up4(bn); c5 = self.conv4(torch.cat([u4, c4], dim=1))
        u3 = self.up3(c5); c6 = self.conv3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(c6); c7 = self.conv2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(c7); c8 = self.conv1(torch.cat([u1, c1], dim=1))
        return self.out(c8)