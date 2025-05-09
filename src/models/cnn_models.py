import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HighAccuracy3DCNN(nn.Module):
    def __init__(self):
        super(HighAccuracy3DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.feature_dim = 64 * 6 * 7 * 6 
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ResNet3DCNN(nn.Module):
    """3‑D CNN tailored for 97×115×97 volumes.
    Uses stride‑2 valid (zero‑padded) convolutions and residual links.
    No average pooling – we flatten the final feature map directly.
    """

    def __init__(self):
        super(ResNet3DCNN, self).__init__()

        # --------------------------------------------------
        # Convolutional backbone (stride = 2, padding = 0)
        # --------------------------------------------------
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=7, stride=4, padding=0),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True)
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(2, 4, kernel_size=5, stride=2, padding=0),
        #     nn.BatchNorm3d(4),
        #     nn.ReLU(inplace=True)
        # )

        #self.conv3 = nn.Sequential(
        #    nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=0),
        #    nn.BatchNorm3d(8),
        #    nn.ReLU(inplace=True)
        #)

        #self.conv4 = nn.Sequential(
        #    nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=0),
        #    nn.BatchNorm3d(16),
        #    nn.ReLU(inplace=True)
        #)

        # --------------------------------------------------
        # Residual down-sampling paths (match size & channels)
        # --------------------------------------------------
        # self.down1 = nn.Sequential(
        #     nn.Conv3d(2, 4, kernel_size=5, stride=2, padding=0),
        #     nn.BatchNorm3d(4)
        # )

        #self.down2 = nn.Sequential(
        #    nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=0),
        #    nn.BatchNorm3d(8)
        #)

        #self.down3 = nn.Sequential(
        #    nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=0),
        #    nn.BatchNorm3d(16)
        #)

        # --------------------------------------------------
        # Classification head (flatten ➜ FC layers)
        # Output size after conv4 for 97×115×97 input: (B, 64, 4, 5, 4)
        # --------------------------------------------------
        #self.feature_dim = 64 * 4 * 5 * 4 # 5120
        self.feature_dim = 2*23*28*23
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # conv1
        x1 = self.conv1(x)

        # conv2 + residual from conv1
        #x2 = self.conv2(x1)
        #x2 = F.relu(x2 + self.down1(x1))

        # conv3 + residual from conv2
        #x3 = self.conv3(x2)
        #x3 = F.relu(x3 + self.down2(x2))

        # conv4 + residual from conv3
        #x4 = self.conv4(x3)
        #x4 = F.relu(x4 + self.down3(x3))

        # Flatten and classify
        x = x1.view(x1.size(0), -1)
        #x = x4.view(x4.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
