import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # Standard ResNet block: Conv -> BN -> ReLU -> Dropout -> Conv -> BN
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.5 to 0.2

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If input shape != output shape, we adjust the identity (shortcut)
        if self.downsample is not None:
            identity = self.downsample(x)

        # The magic happens here: Add original input to output
        out += identity
        out = self.relu(out)

        return out


class DeepSleepNet(nn.Module):
    def __init__(self, n_classes=5):
        super(DeepSleepNet, self).__init__()

        # branch 1: Small Filter = Temporal
        self.small_cnn = nn.Sequential(
            # Initial large conv
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.2),  # Reduced from 0.5 to 0.2

            # residual Blocks
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),

            nn.MaxPool1d(4, stride=4)
        )

        # branch 2: Large Filter = Frequency
        self.large_cnn = nn.Sequential(
            # Initial massive conv
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=200, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(0.2),  # Reduced from 0.5 to 0.2

            # residual blocks
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),

            nn.MaxPool1d(2, stride=2)
        )

        self.dropout = nn.Dropout(0.3)  # Reduced from 0.5 to 0.3

        # automatic feature size calculation = We pass a dummy input to calculate exact output size
        dummy_input = torch.zeros(1, 1, 3000)
        small_out = self.small_cnn(dummy_input)
        large_out = self.large_cnn(dummy_input)

        self.features_dim = small_out.numel() + large_out.numel()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.features_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.fc = nn.Linear(512 * 2, n_classes)

    def _make_layer(self, in_channels, out_channels, stride=1):
        # Helper to create a Residual Block
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        return ResBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x):
        # x shape: (Batch, Seq, 1, 3000)
        batch_size, seq_len, C, T = x.size()

        # Reshape for CNN
        x = x.view(batch_size * seq_len, C, T)

        # Feature Extraction
        small_out = self.small_cnn(x)
        small_out = small_out.view(small_out.size(0), -1)

        large_out = self.large_cnn(x)
        large_out = large_out.view(large_out.size(0), -1)

        # Combine
        features = torch.cat((small_out, large_out), dim=1)
        features = self.dropout(features)

        # Reshape for LSTM
        features = features.view(batch_size, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(features)

        # Classifier
        lstm_out = lstm_out.reshape(batch_size * seq_len, -1)
        return self.fc(lstm_out)
