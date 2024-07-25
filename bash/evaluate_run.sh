#!/bin/bash

# Define the paths to your predictions, ground truth, and log CSV
PREDICTIONS_PATH="/home/guoj5/Desktop/200_annotation/temp_724/cellvit/png"
GT_PATH="/home/guoj5/Documents/200_annotation_all/annotation_mask"
LOG_CSV_PATH="/home/guoj5/Desktop/200_annotation/temp_724/cellvit.csv"

# Run the Python evaluation script with the specified arguments
python cell_segmentation/evaluate.py --predictions "$PREDICTIONS_PATH" --gt "$GT_PATH" --log_csv "$LOG_CSV_PATH"
