#!/bin/bash

python create_dataset.py config.yml state_dicts/ \
    da-vinci ensemble-dataset/ --home ../ \
    --batch-size 4 --save-images ensemble-images/