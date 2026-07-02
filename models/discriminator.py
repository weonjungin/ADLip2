#/home/ihjung/2026/ADLip2/models/discriminator.py
import torch.nn as nn
import torch.nn.functional as F


class MouthDiscriminator(nn.Module):
    def __init__(self, ndf: int = 32, in_ch: int = 5):  # 4ch latent + 1ch mel
        super().__init__()

        # 12×12 입력 → stride=1로 spatial 유지
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 3, 1, 1),      # 12×12
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 3, 1, 1),       # 12×12
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 3, 1, 1),     # 12×12
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Conv2d(ndf*4, 1, 3, 1, 1)   # 12×12 PatchGAN output

    def forward(self, x):
        out, _ = self.get_features(x)
        return out

    def get_features(self, x):
        feats = []
        x = self.conv1(x); feats.append(x)
        x = self.conv2(x); feats.append(x)
        x = self.conv3(x); feats.append(x)
        return self.out(x), feats