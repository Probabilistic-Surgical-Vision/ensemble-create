# Ensemble Dataset Creator

This repository holds the code for creating a ground-truth stereo depth and uncertainty dataset from an ensemble of monocular depth estimation models. This code is used to create a mean depth map, and estimate the per-pixel variance of the Hamlyn and SCARED datasets, in order to train the teacher student model.

## Pre-requisites and installation

To use this package, you will need Python 3.6 or higher. Using an NVIDIA GPU, such as an RTX6000 is recommended.

Download the repository from GitHub and create a virtual environment and activate it:
```bash
python -m venv venv
. venv/bin/activate
```

Install all the packages from pip
```bash
python -m pip install -r requirements.txt
```

## Usage

To use the package, you can either:
- Import the `create` package and use it in your own python code.
- Run `create_dataset.py`
- Run the demo script `create_dataset.sh`

The script will load each state dictionary (identified as any `.pt` file in `state_dicts`) and get its predictions for the train and test datasets. The mean across all model predictions for a single is calculated as the ground truth, and the variance is calculated for the uncertainty. The mean and variance are channel-wise concatenated, and converted to 32-bit float numpy arrays. These are stored as `.tiff` files in the location in the arguments.

## Results from the Ensemble Dataset

![Ensemble dataset in action](./ensemble-mini.gif)