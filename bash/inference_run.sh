#!/bin/bash

# Define the arguments for the inference script
GPU=0
MODEL_PATH="/home/guoj5/Desktop/cellvit/CellViT-kidney/models/pretrained/CellViT-256-x40.pth"
PATCHING=True
OVERLAP=0
DATASETS_DIR="/home/guoj5/Documents/200_annotation_all/png"
OUTPUTS_DIR="/home/guoj5/Desktop/200_annotation/temp_724/cellvit"

# Run the Python inference script with the specified arguments
python cell_segmentation/inference/inference_cellvit_experiment_kidney.py --gpu $GPU --model "$MODEL_PATH" --patching $PATCHING --overlap $OVERLAP --datasets_dir "$DATASETS_DIR" --outputs_dir "$OUTPUTS_DIR"
