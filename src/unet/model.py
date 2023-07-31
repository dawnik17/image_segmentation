import torch
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


"""
upsampling blocks 
(second half of the 'U' in UNet) 
[DECODER]
"""


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


class UNet(nn.Module):
    def __init__(self, encoder_config, decoder_config, nclasses):
        super(UNet, self).__init__()
        self.encoder = Encoder(config=encoder_config)
        self.decoder = Decoder(config=decoder_config)

        self.output = nn.Conv2d(
            in_channels=decoder_config["block1"]["out_channels"],
            out_channels=nclasses,
            kernel_size=1,
        )

    def forward(self, x):
        x, encoder_step_output = self.encoder(x)
        x = self.decoder(x, encoder_step_output)
        return self.output(x)
