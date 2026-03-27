import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y

class NLDMBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(NLDMBlock, self).__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(inchannels, outchannels),
            nn.ReLU(),
            DepthwiseSeparableConv(outchannels, outchannels),
            SEBlock(outchannels),
            nn.MaxPool3d(2, 2)
        )

    def forward(self, x):
        new_features = self.block(x)
        x = F.avg_pool3d(x, 2)
        return torch.cat([x, new_features], 1)

class superLPNet(nn.Module):
    def __init__(self, nb_filter=5, nb_block=5, use_gender=True):
        super(superLPNet, self).__init__()
        self.use_gender = use_gender

        # NCEM: Neuro Context Encoding Module
        self.NCEM = nn.Sequential(
            nn.Conv3d(1, nb_filter, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
        )

        # NLDM: NeuroLite Dense Module
        self.NLDM, last_channels = self._make_NLDM(nb_filter, nb_block)

        # GAP 
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        # CFAM: Compact Feature Aggregation Module
        self.CFAM = nn.Sequential(
            nn.Linear(last_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def _make_NLDM(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(NLDMBlock(inchannels, outchannels))
            inchannels = inchannels + outchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x, a):
        x = self.NCEM(x)
        x = self.NLDM(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.CFAM(x)
        return x
