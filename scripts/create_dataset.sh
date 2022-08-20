#!/bin/bash

. venv/bin/activate

python create_dataset.py config.yml state_dicts/ da-vinci \
    ensemble-dataset/ --save-images ensemble-images/ \
    --batch-size 8 --home ../ --split train