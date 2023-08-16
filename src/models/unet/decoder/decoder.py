import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=2, stride=2, padding=[0, 0]
    ):
        super(DecoderLayer, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding[0],
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv = nn.Sequential(
            *[
                self._conv_relu_layer(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    padding=padding[1],
                )
                for i in range(2)
            ]
        )

    def _conv_relu_layer(self, in_channels, out_channels, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=padding,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    @staticmethod
    def crop_cat(x, encoder_output):
        delta = (encoder_output.shape[-1] - x.shape[-1]) // 2
        encoder_output = encoder_output[
            :, :, delta : delta + x.shape[-1], delta : delta + x.shape[-1]
        ]
        return torch.cat((encoder_output, x), dim=1)

    def forward(self, x, encoder_output):     
        x = self.crop_cat(self.up_conv(x), encoder_output)
        x = self.bn1(x)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleDict(
            {
                name: DecoderLayer(
                    in_channels=block["in_channels"],
                    out_channels=block["out_channels"],
                    kernel_size=block["kernel_size"],
                    stride=block["stride"],
                    padding=block["padding"],
                )
                for name, block in config.items()
            }
        )

    def forward(self, x, encoder_output):
        for name, block in self.decoder.items():
            x = block(x, encoder_output[name]) 
        return x