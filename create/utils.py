from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import Tensor

from torchvision import transforms

ImageDict = Dict[str, Tensor]
ImageSize = Tuple[int, int]
Device = Union[str, torch.device]


class ResizeImage:
    """Resize the stereo images grouped in a dictionary.

    Args:
        size (ImageSize, optional): The target image size. Defaults to
            (256, 512).
    """
    def __init__(self, size: ImageSize = (256, 512)) -> None:
        self.transform = transforms.Resize(size)
        self.size = size

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        image_pair['left'] = self.transform(image_pair['left'])
        image_pair['right'] = self.transform(image_pair['right'])

        if 'ensemble' in image_pair:
            ensemble = image_pair['ensemble'].unsqueeze(0)
            ensemble = F.interpolate(ensemble, self.size, mode='bilinear',
                                     align_corners=True)

            image_pair['ensemble'] = ensemble.squeeze(0)

        return image_pair


class ToTensor:
    """Convert stereo PIL images grouped in a dictionary."""
    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        image_pair['left'] = self.transform(image_pair['left'])
        image_pair['right'] = self.transform(image_pair['right'])

        return image_pair


def to_heatmap(x: Tensor, device: Device = 'cpu', inverse: bool = False,
               colour_map: str = 'inferno') -> Tensor:
    """Convert a single-channel image to an RGB heatmap.

    Args:
        x (Tensor): The single-channel image to convert.
        device (Device, optional): The torch device to map the output to.
            Defaults to 'cpu'.
        inverse (bool, optional): Reverse the heatmap colours. Defaults to
            False.
        colour_map (str, optional): The matpltlib colour map to convert to.
            Defaults to 'inferno'.

    Returns:
        Tensor: The single-channel image as an RGB image.
    """
    image = x.squeeze(0).cpu().numpy()
    image = 1 - image if inverse else image

    transform = plt.get_cmap(colour_map)
    heatmap = transform(image)[:, :, :3]  # remove alpha channel

    return torch.from_numpy(heatmap).to(device).permute(2, 0, 1)