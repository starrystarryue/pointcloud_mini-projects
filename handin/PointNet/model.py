import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """Point-wise MLP implemented as 1x1 convolution."""

    def __init__(self, in_channels, out_channels, use_bn=True, activation=True):
        super().__init__()
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TNet(nn.Module):
    """Transformation network used by PointNet.

    It predicts a k x k affine transform and is initialized as identity,
    matching the original PointNet implementation.
    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = SharedMLP(k, 64)
        self.conv2 = SharedMLP(64, 128)
        self.conv3 = SharedMLP(128, 1024)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x: [B, k, N]
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, dim=2)[0]

        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.fc3(x)

        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k)
        x = x + identity
        return x.view(batch_size, self.k, self.k)


class PointNetEncoder(nn.Module):
    """PointNet encoder for classification."""

    def __init__(self, input_channels=3, feature_transform=True):
        super().__init__()
        self.feature_transform_enabled = feature_transform

        self.input_transform = TNet(k=input_channels)
        self.mlp1 = SharedMLP(input_channels, 64)
        self.mlp2 = SharedMLP(64, 64)

        self.feature_transform = TNet(k=64) if feature_transform else None

        self.mlp3 = SharedMLP(64, 64)
        self.mlp4 = SharedMLP(64, 128)
        self.mlp5 = SharedMLP(128, 1024)

    def forward(self, points):
        # points: [B, N, C]
        x = points.transpose(1, 2).contiguous()
        trans_input = self.input_transform(x)
        x = torch.bmm(points, trans_input).transpose(1, 2).contiguous()

        x = self.mlp1(x)
        x = self.mlp2(x)

        trans_feat = None
        if self.feature_transform is not None:
            trans_feat = self.feature_transform(x)
            x = torch.bmm(x.transpose(1, 2).contiguous(), trans_feat)
            x = x.transpose(1, 2).contiguous()

        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        global_features = torch.max(x, dim=2)[0]
        return global_features, trans_input, trans_feat


class PointNetCls(nn.Module):
    """PointNet classification network for ModelNet40.

    Original classification architecture:
    MLP(64, 64) -> feature transform -> MLP(64, 128, 1024)
    -> max pool -> FC(512, 256, num_classes), with dropout after
    the first two fully connected layers.
    """

    def __init__(self, num_classes=40, input_channels=3, feature_transform=True, dropout=0.3):
        super().__init__()
        self.encoder = PointNetEncoder(
            input_channels=input_channels,
            feature_transform=feature_transform,
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        x, trans_input, trans_feat = self.encoder(points)
        logits = self.classifier(x)
        return logits, trans_input, trans_feat


class PointNet(nn.Module):
    """Compatibility wrapper for classification scripts."""

    def __init__(
        self,
        num_classes=40,
        input_channels=3,
        feature_transform=True,
        dropout=0.3,
    ):
        super().__init__()
        self.net = PointNetCls(num_classes, input_channels, feature_transform, dropout)

    def forward(self, points):
        return self.net(points)


def feature_transform_regularizer(trans):
    """Orthogonality regularizer used in PointNet.

    Use the squared Frobenius distance to avoid the undefined gradient of
    sqrt at exactly zero when the transform is initialized as identity.
    """

    if trans is None:
        return torch.tensor(0.0)
    k = trans.size(1)
    identity = torch.eye(k, device=trans.device, dtype=trans.dtype).unsqueeze(0)
    product = torch.bmm(trans, trans.transpose(1, 2))
    diff = product - identity
    return torch.mean(torch.sum(diff * diff, dim=(1, 2)))
