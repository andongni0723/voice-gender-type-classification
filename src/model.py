from torch import nn


class VoiceClassifyModel(nn.Module):
    def __init__(self, classify_class):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        i = 16
        self.adaptive_pool = nn.AdaptiveAvgPool2d((i, i))
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(in_features=64 * i * i, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=classify_class)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x