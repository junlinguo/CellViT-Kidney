#!/bin/bash

# Define the path to the script and the common arguments
SCRIPT_PATH="/home/guoj5/Desktop/cellvit/CellViT-kidney/cell_segmentation/run_cellvit_mod.py"
GPU="0"

# Define the experiment configs for each experiment
CONFIG1="/home/guoj5/Desktop/cellvit/CellViT-kidney/configs/examples/cell_segmentation/my_experiment_logs/train_cellvit/train_fold5.yaml"
#CONFIG2="/home/guoj5/Desktop/cellvit/CellViT-kidney/configs/examples/cell_segmentation/my_experiment_logs/train_cellvit/train_fold3.yaml"

# Run the first experiment
python $SCRIPT_PATH --gpu $GPU --config $CONFIG1

# Run the second experiment
#python $SCRIPT_PATH --gpu $GPU --config $CONFIG2
