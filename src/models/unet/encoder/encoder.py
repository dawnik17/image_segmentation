import torch.nn as nn


"""
downsampling blocks 
(first half of the 'U' in UNet) 
[ENCODER]
"""


class EncoderLayer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        n_layers=2,
        all_padding=False,
        maxpool=True,
    ):
        super(EncoderLayer, self).__init__()

        f_in_channel = lambda layer: in_channels if layer == 0 else out_channels
        f_padding = lambda layer: 1 if layer >= 2 or all_padding else 0

        self.layer = nn.Sequential(
            *[
                self._conv_relu_layer(
                    in_channels=f_in_channel(i),
                    out_channels=out_channels,
                    padding=f_padding(i),
                )
                for i in range(n_layers)
            ]
        )
        self.maxpool = maxpool

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

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleDict(
            {
                name: EncoderLayer(
                    in_channels=block["in_channels"],
                    out_channels=block["out_channels"],
                    n_layers=block["n_layers"],
                    all_padding=block["all_padding"],
                    maxpool=block["maxpool"],
                )
                for name, block in config.items()
            }
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        output = dict()

        for i, (block_name, block) in enumerate(self.encoder.items()):
            x = block(x)
            output[block_name] = x

            if block.maxpool:
                x = self.maxpool(x)

        return x, output