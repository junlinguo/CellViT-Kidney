
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

**Local run training example:**

```bash
python /path/to/run_cellvit_mod.py --gpu 0 --config /path/to/CellViT-kidney/configs/examples/cell_segmentation/my_experiment_logs/train_cellvit/train_fold0_all.yaml
```

### Bash Scripts to Run Experiments in CLI (with wandb online logging)

`/path/to/CellViT-kidney/bash/run_experiment.sh`

### Inference Script
`/path/to/CellViT-kidney/cell_segmentation/inference/inference_cellvit_experiment_kidney.py`


```bash
python /path/to/inference_cellvit_experiment_kidney.py --gpu 0 --model /path/to/model_checkpoint.pth --patching True --overlap 0 --datasets_dir /path/to/png_folder(s) --outputs_dir /path/to/output_folder
```

### Evaluate 

```bash
python /path/to/cell_segmentation/evaluate.py --predictions /path/to/the/prediction/folder --gt /path/to/the/groundth/folder --log_csv[optional] /path/to/save/metric/csv

```
## Appendix 

### Customize 'experiment_cellvit_instance.py' to finetune pretrained model 
- [Loss functiuons: Disable nuclei type classification and tissue classification objectives](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/experiments/experiment_cellvit_instance.py#L392-L420)
- [Get transforms: Reshape/Crop image to (256, 256) to continue finetuning the pretrained model](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/experiments/experiment_cellvit_instance.py#L729)
- Dataloader: drop_last = True
- [select_dataset: adding 'PanNukeDataset_mod'](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/experiments/experiment_cellvit_instance.py#L530-L536)


### Customize 'cell_segmentation/datasets/pannuke_like_instance.py' [class PanNukeDataset_mod]
- [nuclei/tissue types information: disabled](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/datasets/pannuke_like_instance.py#L94-L100)
- [\_\_getitem\_\_( ): return image, masks, tissue_type(as unknown), image_name](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/datasets/pannuke_like_instance.py#L111).
- "single dummy tissue": Adrenal_gland. Just a placeholder and it does not affect the training objectives
- [loading instance mask](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/datasets/pannuke_like_instance.py#L227-L245)
- [masks include: instance_map, nuclei_type_map (set to be the same as nuclei_binary map), nuclei_binary_map, hv_map](https://github.com/junlinguo/CellViT-Kidney/blob/main/cell_segmentation/datasets/pannuke_like_instance.py#L168)


### Customize 'base_ml/base_trainer.py'


### CellViT arch 

`/CellViT-kidney/models/segmentation/cell_segmentation/cellvit.py`


