import torchsnooper
from torch.nn import Module
from torch import nn


class SEBlock(Module):
    def __init__(self, in_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_map):
        bs, ch, _, _ = feature_map.size()
        y = self.avg_pool(feature_map).view(bs, ch)
        y = self.fc(y).view(bs, ch, 1, 1)
        return feature_map * y.expand_as(feature_map)


class SENet(Module):
    def __init__(self, in_channels, reduction, out_features=1024):
        super(SENet, self).__init__()
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            SEBlock(64, reduction),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            SEBlock(64, reduction),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            SEBlock(128, reduction),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            SEBlock(256, reduction),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            SEBlock(512, reduction)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map):
        bs = feature_map.size(0)
        feature_map = self.se(feature_map)
        feature_vector = self.gap(feature_map).view(bs, -1)

        return feature_vector
