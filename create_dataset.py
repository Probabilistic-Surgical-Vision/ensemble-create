import argparse
import os
import os.path

import torch
import yaml

import create
from model import RandomlyConnectedModel

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str,
                    help='The path to the config file.')
parser.add_argument('ensemble', type=str,
                    help='The path to the state dicts to use as an ensemble.')
parser.add_argument('dataset', choices=['da-vinci', 'scared'], type=str,
                    help='The dataset to generate ensemble predictions for.')
parser.add_argument('save', type=str,
                    help='The location to save the ensemble predictions to.')
parser.add_argument('--save-images', type=str, default=None,
                    help='The location to save the ensemble images to.')
parser.add_argument('--batch-size', '-b', default=8, type=int,
                    help='The batch size to train/evaluate the model with.')
parser.add_argument('--split', choices=['train', 'test'], default='train',
                    help='The dataset to generate ensemble predictions for.')
parser.add_argument('--workers', '-w', default=8, type=int,
                    help='The number of workers to use for the dataloader.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Prevent program from training model using cuda.')
parser.add_argument('--home', default=os.environ['HOME'], type=str,
                    help='Override the home directory (to find datasets).')


def main(args: argparse.Namespace) -> None:
    print("Arguments passed:")
    for key, value in vars(args).items():
        print(f'\t- {key}: {value}')

    dataset_path = os.path.join(args.home, 'datasets', args.dataset)

    save = os.path.join(args.save, args.dataset, args.split)
    save_images = os.path.join(args.save_images, args.dataset, args.split)

    device = torch.device('cuda') if torch.cuda.is_available() \
        and not args.no_cuda else torch.device('cpu')

    creator = create.CreateEnsembleDataset(args.ensemble, args.dataset,
                                           dataset_path, args.split,
                                           args.batch_size,
                                           workers=args.workers,
                                           device=device)
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    model = RandomlyConnectedModel(**config).to(device)

    creator.create(model, save, save_images)

    print(f'Ensemble dataset for "{args.dataset}" completed.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)