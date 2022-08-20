import json
import os
import os.path
import glob

from collections import OrderedDict
from typing import Optional

import numpy as np

import tifffile

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import tqdm

from .utils import Device, ImageSize, ResizeImage, ToTensor, to_heatmap
from .loaders import DaVinciFilePathDataset, SCAREDFilePathDataset


class CreateEnsembleDataset(Dataset):

    def __init__(self, models_path: str, dataset: str, 
                 dataset_path: str, split: str = 'train',
                 batch_size: int = 8,
                 image_size: ImageSize = (256, 512),
                 workers: int = 8, device: Device = 'cpu') -> None:

        if dataset not in ('scared', 'da-vinci'):
            raise ValueError('Dataset must be either "scared" or "da-vinci".')

        self.model_states = []
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = device

        models_glob = os.path.join(models_path, '*.pt')

        for model_path in glob.glob(models_glob):
            model_state = self.prepare_state_dict(model_path)

            self.model_states.append(model_state)

        transform = transforms.Compose([
            ResizeImage(image_size),
            ToTensor()])
    
        dataset_class = DaVinciFilePathDataset \
            if dataset == 'da-vinci' \
            else SCAREDFilePathDataset

        self.dataset = dataset_class(dataset_path, split, transform)
        self.dataloader = DataLoader(self.dataset, batch_size,
                                     shuffle=False, num_workers=workers)
    
    def prepare_state_dict(self, model_path: str) -> OrderedDict:
        state_dict = torch.load(model_path, map_location=self.device)
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    torch.no_grad()
    def ensemble_predict(self, image: Tensor, model: Module) -> Tensor:

        predictions = []

        for state_dict in self.model_states:
            model.load_state_dict(state_dict)
            prediction = model(image)

            predictions.append(prediction)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return torch.cat((mean, variance), dim=1)

    def save_image(self, images: Tensor, prediction: Tensor,
                   filepath: str, scale: bool = True) -> None:
        
        left, right = torch.split(images, [3, 3], dim=0)

        if scale:
            mean, var = torch.split(prediction, [2, 2], dim=0)
            mean = (mean - mean.min()) / (mean.max() - mean.min())
            var = (var - var.min()) / (var.max() - var.min())

            prediction = torch.cat((mean, var), dim=0)

        prediction_heatmap = [left, right]

        for image in prediction:
            depth = to_heatmap(image, self.device)
            prediction_heatmap.append(depth)

        prediction_heatmap = torch.stack(prediction_heatmap)
        image = make_grid(prediction_heatmap, nrow=2)

        save_image(image, filepath)

    @torch.no_grad()
    def create(self, blank_model: Module, save_to: str,
               save_images_to: Optional[str]) -> None:

        os.makedirs(save_to, exist_ok=True)

        if save_images_to is not None:
            os.makedirs(save_images_to, exist_ok=True)

        model = blank_model.to(self.device)
        model.eval()

        uncertainty_means, uncertainty_vars = 0, 0

        for image_pair in tqdm.tqdm(self.dataloader, unit='batch'):
            left = image_pair['left'].to(self.device)
            right = image_pair['right'].to(self.device)

            filenames = image_pair['filename']

            images = torch.cat((left, right), dim=1)
            estimations = self.ensemble_predict(left, model)

            iterable = zip(images, estimations, filenames)

            for image, estimation, filename in iterable:
                uncertainty_means += estimation[2:].mean()
                uncertainty_vars += estimation[2:].var()

                if save_images_to is not None:
                    
                    filepath = os.path.join(save_images_to, f'{filename}.png')
                    self.save_image(image, estimation, filepath)

                filepath = os.path.join(save_to, f'{filename}.tiff')
                estimation = estimation.cpu().numpy().astype(np.float32)

                tifffile.imwrite(filepath, estimation)
            
        normalisation_path = os.path.join(save_to, 'normalisation.json')
        normalisation_dict = {
            'mean': uncertainty_means.item() / len(self.dataset),
            'var': uncertainty_vars.item() / len(self.dataset)
        }

        with open(normalisation_path, 'w+') as f:
            json.dump(normalisation_dict, f, indent=4)