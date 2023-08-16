import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from easydict import EasyDict
from PIL import Image

from src.models.unet.resunet import UNet as Model


class ResUnetInfer:
    def __init__(self, model_path, config_path):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.config = self.load_config(config_path=config_path)
        self.model = self.load_model(model_path=model_path)

        self.transform = A.Compose(
            [
                A.Resize(self.config.input_size[0], self.config.input_size[1]),
                A.Normalize(
                    mean=self.config.mean,
                    std=self.config.std,
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def load_model(self, model_path):
        model = Model(
            decoder_config=self.config.decoder_config, nclasses=self.config.nclasses
        ).to(self.device)

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.decoder.load_state_dict(
                checkpoint["decoder_state_dict"], strict=False
            )
            model.output.load_state_dict(checkpoint["output_state_dict"], strict=False)

        return model

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        return EasyDict(yaml_data)

    def infer(self, image, image_weight=0.01):
        self.model.eval()
        input_tensor = self.transform(image=image)["image"].unsqueeze(0)

        # get mask
        with torch.no_grad():
            """
            output_tensor = [batch, 1, 224, 224]
            batch = 1
            """
            output_tensor = self.model(input_tensor.to(self.device))
        
        mask = torch.sigmoid(output_tensor)
        mask = nn.UpsamplingBilinear2d(size=(image.shape[0], image.shape[1]))(mask)
        mask = mask.squeeze(0)

        # add zeros for green and blue channels
        # our mask will be red in colour
        zero_channels = torch.zeros((2, image.shape[0], image.shape[1]), device=self.device)
        mask = torch.cat([mask, zero_channels], dim=0)
        mask = mask.permute(1,2,0).cpu().numpy()
        mask = np.uint8(255 * mask)
        
        # overlap image and mask
        mask = (1 - image_weight) * mask + image_weight * image
        mask = mask / np.max(mask)
        return np.uint8(255 * mask)
    
    @staticmethod
    def load_image_as_array(image_path):
        # Load a PIL image
        pil_image = Image.open(image_path)

        # Convert PIL image to NumPy array
        return np.array(pil_image.convert("RGB"))

    @staticmethod
    def plot_array(array: np.array, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        plt.imshow(array)
        plt.show()

    @staticmethod
    def save_numpy_as_image(numpy_array, image_path):
        """
        Saves a NumPy array as an image.
        Args:
            numpy_array (numpy.ndarray): The NumPy array to be saved as an image.
            image_path (str): The path where the image will be saved.
        """
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(numpy_array)

        # Save the PIL image to the specified path
        image.save(image_path)

