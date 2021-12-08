import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, relu=True):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(output_dim)

        if relu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        return self.activation(output)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ResidualBlock, self).__init__()

        self.block_1 = CNNBlock(input_dim, output_dim, kernel_size, relu=True)
        self.block_2 = CNNBlock(output_dim, output_dim, kernel_size, relu=False)

        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = self.residual(x)

        output = self.block_1(x)
        output = self.block_2(output)

        return residual + output


class UpsampleBlock(nn.Sequential):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(UpsampleBlock, self).__init__(
            CNNBlock(input_dim, output_dim, kernel_size, relu=True),
            nn.Conv2d(output_dim, output_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )

        self.output_dim = output_dim // 4


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.initial_norm = nn.BatchNorm2d(3)
        self.initial_embedding = CNNBlock(3, 64, 7)

        self.encoder_1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 128, 3)
        )

        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 128, 3),
            ResidualBlock(128, 128, 3),
            ResidualBlock(128, 128, 3),
            ResidualBlock(128, 256, 3)
        )

        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(256, 256, 3),
            ResidualBlock(256, 256, 3),
            ResidualBlock(256, 256, 3),
            ResidualBlock(256, 256, 3),
            ResidualBlock(256, 256, 3),
            ResidualBlock(256, 512, 3)
        )

        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(512, 512, 3),
            ResidualBlock(512, 512, 3),
            CNNBlock(512, 512, 1),
        )

        self.encoder_5 = nn.Sequential(
            CNNBlock(512, 1024, 3),
            CNNBlock(1024, 512, 3),
        )

        self.decoder_5 = nn.PixelShuffle(1)
        self.decoder_4 = UpsampleBlock(1024, 512, 3)
        self.decoder_3 = UpsampleBlock(512 + 128, 384, 3)
        self.decoder_2 = UpsampleBlock(384 // 4 + 256, 256, 3)
        self.decoder_1 = UpsampleBlock(256 // 4 + 128, 96, 3)

        self.final_decoder = nn.Sequential(
            nn.Conv2d(96 // 4 + 64, 99, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(99, 99, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(99, 3, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_norm(x)
        hidden_1 = self.initial_embedding(x)

        hidden_2 = self.encoder_1(hidden_1)
        hidden_3 = self.encoder_2(hidden_2)
        hidden_4 = self.encoder_3(hidden_3)
        hidden_5 = self.encoder_4(hidden_4)
        hidden_6 = self.encoder_5(hidden_5)

        decoded_5 = self.decoder_5(hidden_6)
        hidden_5 = torch.concat((decoded_5, hidden_5), dim=1)

        decoded_4 = self.decoder_4(hidden_5)
        hidden_4 = torch.concat((decoded_4, hidden_4), dim=1)

        decoded_3 = self.decoder_3(hidden_4)
        hidden_3 = torch.concat((decoded_3, hidden_3), dim=1)

        decoded_2 = self.decoder_2(hidden_3)
        hidden_2 = torch.concat((decoded_2, hidden_2), dim=1)

        decoded_1 = self.decoder_1(hidden_2)
        hidden_1 = torch.concat((decoded_1, hidden_1), dim=1)

        hidden_states = [decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, hidden_6]
        # hidden_states = []

        return self.final_decoder(hidden_1), hidden_states
