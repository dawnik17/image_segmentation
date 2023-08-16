import torch.nn as nn
from .encoder import ResnetEncoder as Encoder
from .decoder import CustomDecoder as Decoder


class UNet(nn.Module):
    def __init__(self, decoder_config, nclasses, input_shape=(224, 224)):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(config=decoder_config)
        
        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_config["block1"]["out_channels"],
                out_channels=nclasses,
                kernel_size=1,
            ),
            nn.UpsamplingBilinear2d(size=input_shape)
        )

    def forward(self, x):
        x, encoder_step_output = self.encoder(x)
        x = self.decoder(x, encoder_step_output)
        x = self.output(x)
        return x
    

if __name__ == "__main__":
    import torch
    import yaml
    from easydict import EasyDict
    from torchinfo import summary

    # load config
    config_path = './config/resnet_config.yml'

    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    config = EasyDict(yaml_data)

    # input shape
    input_shape=(224, 224)

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # model definition
    model = UNet(decoder_config=config["decoder_config"], 
                 nclasses=1, 
                 input_shape=input_shape).to(device)

    summary(
        model,
        input_data=torch.rand((1, 3, input_shape[0], input_shape[1])),
        device=device
    )

    # load weights (if any)
    model_path = None

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=False)
        model.output.load_state_dict(checkpoint["output_state_dict"], strict=False)