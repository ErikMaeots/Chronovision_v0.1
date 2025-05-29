import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        xp = self.global_pool(x)
        xp = self.conv_pool(xp)
        xp = F.interpolate(xp, size=(h, w), mode='bilinear', align_corners=False)
        out = torch.cat([x1, x2, x3, x4, xp], dim=1)
        return self.project(out)

class DecoderWithSkips(nn.Module):
    def __init__(self, in_channels=256, skip_channels=(256, 512, 1024), out_channels=1):
        super(DecoderWithSkips, self).__init__()
        self.reduce4 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.reduce3 = nn.Conv2d(skip_channels[2], 256, kernel_size=1)
        self.reduce2 = nn.Conv2d(skip_channels[1], 256, kernel_size=1)
        self.reduce1 = nn.Conv2d(skip_channels[0], 256, kernel_size=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Extra progressive steps for final upsampling 32→64→128
        self.up_conv64 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up_conv128 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x_aspp, skip1, skip2, skip3):
        # x_aspp = [B, 256, 4, 4]
        # skip3  = [B,1024, 8, 8]
        # skip2  = [B, 512,16,16]
        # skip1  = [B, 256,32,32]

        x = self.reduce4(x_aspp)                                 # [B,256,4,4]
        x = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=False)  # 4->8
        x = x + self.reduce3(skip3)                              # [B,256,8,8]
        x = self.conv3(x)                                        # [B,256,8,8]

        x = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)  # 8->16
        x = x + self.reduce2(skip2)                              # [B,256,16,16]
        x = self.conv2(x)                                        # [B,256,16,16]

        x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)  # 16->32
        x = x + self.reduce1(skip1)                              # [B,256,32,32]
        x = self.conv1(x)                                        # [B,256,32,32]

        # ---- Progressive steps from 32->64->128 to avoid aliasing ----
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)        # 32->64
        x = self.up_conv64(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)        # 64->128
        x = self.up_conv128(x)
        # --------------------------------------------------------------

        return self.final_conv(x)

class DeepLabV3Custom_v3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DeepLabV3Custom_v3, self).__init__()
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        returned_layers = {
            'layer1': 'skip1',
            'layer2': 'skip2',
            'layer3': 'skip3',
            'layer4': 'out'
        }
        self.encoder = IntermediateLayerGetter(backbone, return_layers=returned_layers)
        self.aspp = ASPP(in_channels=2048, out_channels=256, dilations=(6, 12, 18))
        self.decoder = DecoderWithSkips(
            in_channels=256,
            skip_channels=(256, 512, 1024),
            out_channels=out_channels
        )
    def forward(self, x):
        feats = self.encoder(x)
        skip1 = feats['skip1']
        skip2 = feats['skip2']
        skip3 = feats['skip3']
        x_out = feats['out']
        x_aspp = self.aspp(x_out)
        return self.decoder(x_aspp, skip1, skip2, skip3)