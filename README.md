
# CellViT Experiments

## Finetune CellViT256 (HVNet Decoder) on PanNuke Dataset (Paper Experiment)

**Objective:**  
- Nuclei instance segmentation
- Nuclei classification
- Tissue classification

**Command to run:**
```bash
python /path/to/cellvit/CellViT-kidney/cell_segmentation/run_cellvit.py --gpu 0 --config /home/guoj5/Desktop/cellvit/CellViT-kidney/configs/examples/cell_segmentation/train_cellvit_copy.yaml
```

## My Experiments

### CellViT Backbone Experiments

**Training/Finetuning Objective:**  
- Nuclei instance segmentation

**Data:**  
The data are from the previously curated instance-segmentation data.  
Data preparation (as PanNuke dataset file structures, currently) follows: `/mnt/Data/guoj5/fintuned_dummy/instructions.txt`
(or `./dataset_instructions.txt`)

**Main script to run the experiment:**  
`/path/to/CellViT-kidney/cell_segmentation/run_cellvit_mod.py`

**Experiments Configuration [Important!!!]:**  
`/path/to/CellViT-kidney/configs/examples/cell_segmentation/my_experiment_logs/train_cellvit/xxx.yaml`

The experiments are also logged by Weights and Biases.

**Local run example:**
```bash
python /path/to/run_cellvit_mod.py --gpu 0 --config /path/to/CellViT-kidney/configs/examples/cell_segmentation/my_experiment_logs/train_cellvit/train_fold0_all.yaml
```

### Bash Scripts to Run Experiments in CLI (with wandb online logging)

`/path/to/CellViT-kidney/bash/run_experiment.sh`

### Inference Script
`/path/to/CellViT-kidney/cell_segmentation/inference/inference_cellvit_experiment_kidney.py`

## Appendix 
### Customize 'experiment_cellvit_instance.py' to finetune pretrained model 
- [Disable nuclei type classification and tissue classification objectives](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/experiments/experiment_cellvit_instance.py#L392-420)
- [Reshape/Crop image to (256, 256) to continue finetuning the pretrained model](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/experiments/experiment_cellvit_instance.py#L729)