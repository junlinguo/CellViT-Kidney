# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on MoNuSeg dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

# Modified by JUNLIN GUO (Vanderbilt University)
# CellViT Inference for Patch-Wise Inference on private/public unlabeled Kidney dataset
# customize the non-label dataset (cell_segmentation/datasets/own_nonlabel_update.py)

import argparse
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

from pathlib import Path
from typing import List, Union, Tuple

import albumentations as A
import cv2 as cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgba2rgb
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms
import tqdm
import pickle

# Need to modify
from cell_segmentation.datasets.own_nonlabel_update import MoNuSegDataset
from cell_segmentation.inference.cell_detection import (
    CellPostProcessor,
    get_cell_position,
    get_cell_position_marging,
    get_edge_patch,
)
from cell_segmentation.utils.metrics import (
    cell_detection_scores,
    get_fast_pq,
    remap_label,
)
from cell_segmentation.utils.post_proc_cellvit import calculate_instances
from cell_segmentation.utils.tools import pair_coordinates
from models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViT256,
    CellViTSAM,
)
from models.segmentation.cell_segmentation.cellvit_shared import (
    CellViT256Shared,
    CellViTSAMShared,
    CellViTShared,
)
from utils.logger import Logger
from utils.tools import unflatten_dict
from typing import Set

def find_files(directory: str, format: str = '.svs') -> Set[str]:
    """
    Find all folders/subdirectories that contain files with the specified format.
    :param directory: The root directory to search for files.
    :param format: The type of the files to search for (default is '.svs', for the WSI file search).
    :return: A set of directories containing files with the specified format
    """
    svs_directories = set()  # Use a set to store unique parent directories

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(format):
                svs_directories.add(root)  # Use add() to add unique directories

    return svs_directories

class MoNuSegInference:
    def __init__(
        self,
        model_path: Union[Path, str],
        dataset_path: Union[Path, str],
        outdir: Union[Path, str],
        gpu: int,
        patching: bool = False,
        overlap: int = 0,
        magnification: int = 40,
    ) -> None:
        """Cell Segmentation Inference class for MoNuSeg dataset

        Args:
            model_path (Union[Path, str]): Path to model checkpoint
            dataset_path (Union[Path, str]): Path to dataset
            outdir (Union[Path, str]): Output directory
            gpu (int): CUDA GPU id to use
            patching (bool, optional): If dataset should be pacthed to 256px. Defaults to False.
            overlap (int, optional): If overlap should be used. Recommed (next to no overlap) is 64 px. Overlap in px.
                If overlap is used, patching must be True. Defaults to 0.
            magnification (int, optional): Dataset magnification. Defaults to 40.
        """
        self.model_path = Path(model_path)
        self.device = f"cuda:{gpu}"
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.magnification = magnification
        self.overlap = overlap
        self.patching = patching
        if overlap > 0:
            assert patching, "Patching must be activated"

        self.__instantiate_logger()
        self.__load_model()
        self.__load_inference_transforms()
        self.__setup_amp()
        self.inference_dataset = MoNuSegDataset(
            dataset_path=dataset_path,
            transforms=self.inference_transforms,
            patching=patching,
            overlap=overlap,
        )
        self.inference_dataloader = DataLoader(
            self.inference_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

    def __instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
            log_dir=self.outdir,
            comment="inference_monuseg",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __load_model(self) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {self.model_path}")

        model_checkpoint = torch.load(self.model_path, map_location="cpu")

        # unpack checkpoint
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        self.model = self.__get_model(model_type=model_checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)

    def __get_model(
        self, model_type: str
    ) -> Union[
        CellViT,
        CellViTShared,
        CellViT256,
        CellViT256Shared,
        CellViTSAM,
        CellViTSAMShared,
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared

        Returns:
            Union[CellViT, CellViTShared, CellViT256, CellViTShared, CellViTSAM, CellViTSAMShared]: Model
        """
        implemented_models = [
            "CellViT",
            "CellViTShared",
            "CellViT256",
            "CellViT256Shared",
            "CellViTSAM",
            "CellViTSAMShared",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT", "CellViTShared"]:
            if model_type == "CellViT":
                model_class = CellViT
            elif model_type == "CellViTShared":
                model_class = CellViTShared
            model = model_class(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )

        elif model_type in ["CellViT256", "CellViT256Shared"]:
            if model_type == "CellViT256":
                model_class = CellViT256
            elif model_type == "CellViT256Shared":
                model_class = CellViT256Shared
            model = model_class(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM", "CellViTSAMShared"]:
            if model_type == "CellViTSAM":
                model_class = CellViTSAM
            elif model_type == "CellViTSAMShared":
                model_class = CellViTSAMShared
            model = model_class(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        return model

    def __load_inference_transforms(self) -> None:
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = A.Compose([A.Normalize(mean=mean, std=std)])

    def __setup_amp(self) -> None:
        """Setup automated mixed precision (amp) for inference."""
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)

    def run_inference(self, generate_plots: bool = False) -> None:
        """Run inference

        Args:
            generate_plots (bool, optional): If plots should be generated. Defaults to False.
        """
        self.model.eval()

        # setup score tracker
        image_names = []  # image names as str
        binary_dice_scores = []  # binary dice scores per image
        binary_jaccard_scores = []  # binary jaccard scores per image
        pq_scores = []  # pq-scores per image
        dq_scores = []  # dq-scores per image
        sq_scores = []  # sq-scores per image
        f1_ds = []  # f1-scores per image
        prec_ds = []  # precision per image
        rec_ds = []  # recall per image

        inference_loop = tqdm.tqdm(
            enumerate(self.inference_dataloader), total=len(self.inference_dataloader)
        )

        with torch.no_grad():
            for image_idx, batch in inference_loop:

                ######## Added by Junlin, skip the corrupted/truncated image
                if torch.all(batch[0] == 0):
                    continue
                image_metrics = self.inference_step(
                    model=self.model, batch=batch, generate_plots=generate_plots
                )
                ######## Added by JUNLIN
                if image_metrics is None:
                    continue

                image_names.append(image_metrics["image_name"])
                binary_dice_scores.append(image_metrics["binary_dice_score"])
                binary_jaccard_scores.append(image_metrics["binary_jaccard_score"])
                pq_scores.append(image_metrics["pq_score"])
                dq_scores.append(image_metrics["dq_score"])
                sq_scores.append(image_metrics["sq_score"])
                f1_ds.append(image_metrics["f1_d"])
                prec_ds.append(image_metrics["prec_d"])
                rec_ds.append(image_metrics["rec_d"])

        ############# no data stats/metrics when gt is not provided # ##


    def inference_step(
        self, model: nn.Module, batch: object, generate_plots: bool = False
    ) -> dict:
        """Inference step

        Args:
            model (nn.Module): Training model, must return "nuclei_binary_map", "nuclei_type_map", "tissue_type" and "hv_map"
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            generate_plots (bool, optional): If plots should be generated. Defaults to False.

        Returns:
            Dict: Image_metrics with keys:

        """
        img = batch[0].to(self.device)
        if len(img.shape) > 4:
            img = img[0]
            img = rearrange(img, "c i j w h -> (i j) c w h")
        mask = batch[1]
        image_name = list(batch[2])

        # len(mask) == 0 , no mask provided
        if len(mask) > 0:
            mask["instance_types"] = calculate_instances(
                torch.unsqueeze(mask["nuclei_binary_map"], dim=0), mask["instance_map"]
            )

        model.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions_ = model.forward(img)
        else:
            predictions_ = model.forward(img)
            #################### predictions_ {dict:4}####################
            #  'tissue_types', 'nuclei_binary_map', 'hv_map', 'nuclei_type_map'
            ###############################################################

        if self.overlap == 0:
            if self.patching:
                predictions_ = self.post_process_patching(predictions_)
            predictions = self.get_cell_predictions(predictions_)
            image_metrics = self.calculate_step_metric(
                predictions=predictions, gt=mask, image_name=image_name
            )

        elif self.patching and self.overlap != 0:
            cell_list = self.post_process_patching_overlap(
                predictions_, overlap=self.overlap
            )

            image_metrics, predictions = self.calculate_step_metric_overlap(
                cell_list=cell_list, gt=mask, image_name=image_name
            )

        # scores = [
        #     float(image_metrics["binary_dice_score"].detach().cpu()),
        #     float(image_metrics["binary_jaccard_score"].detach().cpu()),
        #     image_metrics["pq_score"],
        # ]
        if generate_plots:
            if self.overlap == 0 and self.patching:
                batch_size = img.shape[0]
                num_elems = int(np.sqrt(batch_size))
                img = torch.permute(img, (0, 2, 3, 1))
                img = rearrange(
                    img, "(i j) h w c -> (i h) (j w) c", i=num_elems, j=num_elems
                )
                img = torch.unsqueeze(img, dim=0)
                img = torch.permute(img, (0, 3, 1, 2))
            elif self.overlap != 0 and self.patching:
                h, w = 512, 512
                total_img = torch.zeros((3, h, w))
                decomposed_patch_num = int(np.sqrt(img.shape[0]))
                for i in range(decomposed_patch_num):
                    for j in range(decomposed_patch_num):
                        x_global = i * 256 - i * self.overlap
                        y_global = j * 256 - j * self.overlap
                        total_img[
                            :, x_global : x_global + 256, y_global : y_global + 256
                        ] = img[i * decomposed_patch_num + j]
                img = total_img
                img = img[None, :, :, :]
            self.plot_results(
                img=img,
                predictions=predictions,
                ground_truth=mask,
                img_name=image_name[0],
                outdir=self.outdir,
                # scores=scores,
                scores= None,
                # add
                binary=False
            )



        return image_metrics

    def calculate_step_metric(
        self, predictions: dict, gt: dict, image_name: List[str]
    ) -> dict:
        """Calculate step metric for one MoNuSeg image.

        Args:
            predictions (dict): Necssary keys:
                * instance_map: Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (1, H, W)
                * nuclei_binary_map: Softmax output for binary nuclei branch. Shape: (1, 2, H, W)
                * instance_types: Instance type prediction list.
                    Each list entry stands for one image. Each list entry is a dictionary with the following structure:
                    Main Key is the nuclei instance number (int), with a dict as value.
                    For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
                    contour, type_prob (probability), type (nuclei type). Actually just one list entry, as we expecting batch-size=1 (one image)
            gt (dict): Necessary keys:
                * instance_map
                * nuclei_binary_map
                * instance_types
            image_name (List[str]): Name of the image, list with [str]. List is necessary for backward compatibility

        Returns:
            dict: Image metrics for one MoNuSeg image. Keys are:
                * image_name
                * binary_dice_score
                * binary_jaccard_score
                * pq_score
                * dq_score
                * sq_score
                * f1_d
                * prec_d
                * rec_d
        """
        ######## Added conditional: when mask/gt is not provided #############
        if len(gt) == 0:
            return None
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        instance_maps_gt = gt["instance_map"].detach().cpu()

        pred_binary_map = torch.argmax(predictions["nuclei_binary_map"], dim=1)
        target_binary_map = gt["nuclei_binary_map"].to(self.device)

        cell_dice = (
            dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
            .detach()
            .cpu()
        )
        cell_jaccard = (
            binary_jaccard_index(
                preds=pred_binary_map,
                target=target_binary_map,
            )
            .detach()
            .cpu()
        )
        remapped_instance_pred = remap_label(predictions["instance_map"])
        remapped_gt = remap_label(instance_maps_gt)
        [dq, sq, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)

        # detection scores
        true_centroids = np.array(
            [v["centroid"] for k, v in gt["instance_types"][0].items()]
        )
        pred_centroids = np.array(
            [v["centroid"] for k, v in predictions["instance_types"].items()]
        )
        if true_centroids.shape[0] == 0:
            true_centroids = np.array([[0, 0]])
        if pred_centroids.shape[0] == 0:
            pred_centroids = np.array([[0, 0]])

        if self.magnification == 40:
            pairing_radius = 12
        else:
            pairing_radius = 6
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroids, pred_centroids, pairing_radius
        )
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=paired[:, 0],
            paired_pred=paired[:, 1],
            unpaired_true=unpaired_true,
            unpaired_pred=unpaired_pred,
        )

        image_metrics = {
            "image_name": image_name,
            "binary_dice_score": cell_dice,
            "binary_jaccard_score": cell_jaccard,
            "pq_score": pq,
            "dq_score": dq,
            "sq_score": sq,
            "f1_d": f1_d,
            "prec_d": prec_d,
            "rec_d": rec_d,
        }

        return image_metrics

    def convert_binary_type(self, instance_types: dict) -> dict:
        """Clean nuclei detection from type prediction to binary prediction

        Args:
            instance_types (dict): Dictionary with cells

        Returns:
            dict: Cleaned with just one class
        """
        cleaned_instance_types = {}
        for key, elem in instance_types.items():
            if elem["type"] == 0:
                continue
            else:
                elem["type"] = 0
                cleaned_instance_types[key] = elem

        return cleaned_instance_types

    def get_cell_predictions(self, predictions: dict) -> dict:
        """Reshaping predictions and calculating instance maps and instance types

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (B, H, W, 2)
                * hv_map: Logit output for hv-prediction. Shape: (B, 2, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (B, num_nuclei_classes, H, W)

        Returns:
            dict:
                * nuclei_binary_map: Softmax binary prediction. Shape: (B, 2, H, W
                * nuclei_type_map: Softmax nuclei type map. Shape: (B, num_nuclei_classes, H, W)
                * hv_map: Logit output for hv-prediction. Shape: (B, 2, H, W)
                * tissue_types: Logit tissue prediction output. Shape: (B, num_tissue_classes)
                * instance_map: Instance map, each instance has one integer. Shape: (B, H, W)
                * instance_types: Instance type dict, cleaned. Keys:
                    'bbox', 'centroid', 'contour', 'type_prob', 'type'
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = self.model.calculate_instance_map(
            predictions, magnification=self.magnification
        )
        predictions["instance_types"] = self.convert_binary_type(
            predictions["instance_types"][0]
        )

        return predictions

    def post_process_patching(self, predictions: dict) -> dict:
        """Post-process patching by reassamble (without overlap) stitched predictions to one big image prediction

        Args:
            predictions (dict): Necessary keys:
                * nuclei_binary_map: Logit binary prediction. Shape: (B, 2, 256, 256)
                * hv_map: Logit output for hv-prediction. Shape: (B, 2, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (B, num_nuclei_classes, 256, 256)
        Returns:
            dict: Return elements that have been changed:
                * nuclei_binary_map: Shape: (1, 2, H, W)
                * hv_map: Shape: (1, 2, H, W)
                * nuclei_type_map: (1, num_nuclei_classes, H, W)
        """
        batch_size = predictions["nuclei_binary_map"].shape[0]
        num_elems = int(np.sqrt(batch_size))
        predictions["nuclei_binary_map"] = rearrange(
            predictions["nuclei_binary_map"],
            "(i j) d w h ->d (i w) (j h)",
            i=num_elems,
            j=num_elems,
        )
        predictions["hv_map"] = rearrange(
            predictions["hv_map"],
            "(i j) d w h -> d (i w) (j h)",
            i=num_elems,
            j=num_elems,
        )
        predictions["nuclei_type_map"] = rearrange(
            predictions["nuclei_type_map"],
            "(i j) d w h -> d (i w) (j h)",
            i=num_elems,
            j=num_elems,
        )

        predictions["nuclei_binary_map"] = torch.unsqueeze(
            predictions["nuclei_binary_map"], dim=0
        )
        predictions["hv_map"] = torch.unsqueeze(predictions["hv_map"], dim=0)
        predictions["nuclei_type_map"] = torch.unsqueeze(
            predictions["nuclei_type_map"], dim=0
        )

        return predictions

    def post_process_patching_overlap(self, predictions: dict, overlap: int) -> List:
        """Post processing overlapping cells by merging overlap. Use same merging strategy as for our

        Args:
            predictions (dict): Predictions with necessary keys:
                * nuclei_binary_map: Binary nuclei prediction, Shape: (B, 2, H, W)
                * nuclei_type_map: Nuclei type prediction, Shape: (B, num_nuclei_classes, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
            overlap (int): Used overlap as integer

        Returns:
            List: Cleaned (merged) cell list with each entry beeing one detected cell with dictionary as entries.
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = self.model.calculate_instance_map(
            predictions, magnification=self.magnification
        )
        predictions = self.merge_predictions(predictions, overlap)

        return predictions

    def merge_predictions(self, predictions: dict, overlap: int) -> list:
        """Merge overlapping cell predictions

        Args:
            predictions (dict): Predictions with necessary keys:
                * nuclei_binary_map: Binary nuclei prediction, Shape: (B, 2, H, W)
                * instance_types: Instance type dictionary with cell entries
            overlap (int): Used overlap as integer

        Returns:
            list: Cleaned (merged) cell list with each entry beeing one detected cell with dictionary as entries.
        """
        cell_list = []
        decomposed_patch_num = int(np.sqrt(predictions["nuclei_binary_map"].shape[0]))

        for i in range(decomposed_patch_num):
            for j in range(decomposed_patch_num):
                x_global = i * 256 - i * overlap
                y_global = j * 256 - j * overlap
                patch_instance_types = predictions["instance_types"][
                    i * decomposed_patch_num + j
                ]
                for cell in patch_instance_types.values():
                    if cell["type"] == 0:
                        continue
                    offset_global = np.array([x_global, y_global])
                    centroid_global = cell["centroid"] + np.flip(offset_global)
                    contour_global = cell["contour"] + np.flip(offset_global)
                    bbox_global = cell["bbox"] + offset_global
                    cell_dict = {
                        "bbox": bbox_global.tolist(),
                        "centroid": centroid_global.tolist(),
                        "contour": contour_global.tolist(),
                        "type_prob": cell["type_prob"],
                        "type": cell["type"],
                        "patch_coordinates": [
                            i,  # row
                            j,  # col
                        ],
                        "cell_status": get_cell_position_marging(cell["bbox"], 256, 64),
                        "offset_global": offset_global.tolist(),
                    }
                    if np.max(cell["bbox"]) == 256 or np.min(cell["bbox"]) == 0:
                        position = get_cell_position(cell["bbox"], 256)
                        cell_dict["edge_position"] = True
                        cell_dict["edge_information"] = {}
                        cell_dict["edge_information"]["position"] = position
                        cell_dict["edge_information"]["edge_patches"] = get_edge_patch(
                            position, i, j  # row, col
                        )
                    else:
                        cell_dict["edge_position"] = False
                    cell_list.append(cell_dict)
        self.logger.info(f"Detected cells before cleaning: {len(cell_list)}")
        cell_processor = CellPostProcessor(cell_list, self.logger)
        cleaned_cells = cell_processor.post_process_cells()
        cell_list = [cell_list[idx_c] for idx_c in cleaned_cells.index.values]
        self.logger.info(f"Detected cells after cleaning: {len(cell_list)}")

        return cell_list

    def calculate_step_metric_overlap(
        self, cell_list: List[dict], gt: dict, image_name: List[str]
    ) -> Tuple[dict, dict]:
        """Calculate step metric and return merged predictions for plotting

        Args:
            cell_list (List[dict]): List with cell dicts
            gt (dict): Ground-Truth dictionary
            image_name (List[str]): Image Name as list with just one entry

        Returns:
            Tuple[dict, dict]:
                dict: Image metrics for one MoNuSeg image. Keys are:
                * image_name
                * binary_dice_score
                * binary_jaccard_score
                * pq_score
                * dq_score
                * sq_score
                * f1_d
                * prec_d
                * rec_d
                dict: Predictions, reshaped for one image and for plotting
                * nuclei_binary_map: Shape (1, 2, 1024, 1024) or (1, 2, 1024, 1024)
                * instance_map: Shape (1, 1024, 1024) or or (1, 2, 512, 512)
                * instance_types: Dict for each nuclei
        """
        predictions = {}

        h, w = 512, 512
        instance_type_map = np.zeros((h, w), dtype=np.int32)

        for instance, cell in enumerate(cell_list):
            contour = np.array(cell["contour"])[None, :, :]
            cv2.fillPoly(instance_type_map, contour, instance)

        predictions["instance_map"] = torch.Tensor(instance_type_map)


        pred_arr = np.clip(instance_type_map, 0, 1)
        predictions["nuclei_binary_map"] = pred_arr

        predictions["instance_types"] = cell_list


        remapped_instance_pred = remap_label(predictions["instance_map"])[None, :, :]

        pred_centroids = np.array([v["centroid"] for v in cell_list])

        if pred_centroids.shape[0] == 0:
            pred_centroids = np.array([[0, 0]])

        if self.magnification == 40:
            pairing_radius = 12
        else:
            pairing_radius = 6


        # align to common shapes
        cleaned_instance_types = {
            k + 1: v for k, v in enumerate(predictions["instance_types"])
        }
        for cell, results in cleaned_instance_types.items():
            results["contour"] = np.array(results["contour"])
            cleaned_instance_types[cell] = results
        predictions["instance_types"] = cleaned_instance_types
        predictions["instance_map"] = predictions["instance_map"][None, :, :]
        predictions["nuclei_binary_map"] = F.one_hot(
            torch.Tensor(predictions["nuclei_binary_map"]).type(torch.int64),
            num_classes=2,
        ).permute(2, 0, 1)[None, :, :, :]

        return None, predictions

    def plot_results(
        self,
        img: torch.Tensor,
        predictions: dict,
        ground_truth: dict,
        img_name: str,
        outdir: Path,
        scores: List[float],
        binary = False
    ) -> None:
        """Plot MoNuSeg results

        Args:
            img (torch.Tensor): Image as torch.Tensor, with Shape (1, 3, 1024, 1024) or (1, 3, 512, 512)
            predictions (dict): Prediction dictionary. Necessary keys:
                * nuclei_binary_map: Shape (1, 2, 1024, 1024) or (1, 2, 512, 512)
                * instance_map: Shape (1, 1024, 1024) or (1, 512, 512)
                * instance_types: List[dict], but just one entry in list
            ground_truth (dict): Ground-Truth dictionary. Necessary keys:
                * nuclei_binary_map: (1, 1024, 1024) or or (1, 512, 512)
                * instance_map: (1, 1024, 1024) or or (1, 512, 512)
                * instance_types: List[dict], but just one entry in list
            img_name (str): Image name as string
            outdir (Path): Output directory for storing
            scores (List[float]): Scores as list [Dice, Jaccard, bPQ]
        """

        # outdir = Path(outdir) / "plots" # default
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )

        h = 512
        w = 512

        # process image and other maps
        sample_image = img.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        pred_sample_binary_map = (
            predictions["nuclei_binary_map"][:, :, :, 1].detach().cpu().numpy()
        )[0]
        pred_sample_instance_maps = (
            predictions["instance_map"].detach().cpu().numpy()[0]
        )


        # invert the normalization of the sample images
        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        inv_normalize = transforms.Normalize(
            mean=[-0.5 / mean[0], -0.5 / mean[1], -0.5 / mean[2]],
            std=[1 / std[0], 1 / std[1], 1 / std[2]],
        )
        inv_samples = inv_normalize(torch.tensor(sample_image).permute(0, 3, 1, 2))
        sample_image = inv_samples.permute(0, 2, 3, 1).detach().cpu().numpy()[0]

        ## Binary map
        if binary:
            binary_map = (pred_sample_binary_map.astype(np.uint8)) * 255
            image = Image.fromarray(np.stack((binary_map, binary_map, binary_map), axis=-1))
            image.save(os.path.join(outdir, f'{img_name.replace(".png", "")}_binary.png'))

        ## instance map
        np.save(os.path.join(outdir, f'{img_name.replace(".png", "")}_contours.npy'), pred_sample_instance_maps)

        ## draw contours on the sample image
        mask = pred_sample_instance_maps
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]

        out_img = (sample_image*255).astype(np.uint8)

        for label in unique_labels:
            binary_mask = np.where(mask == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(out_img, contours, -1, (0, 255, 0), 3)

        Image.fromarray(out_img).save(os.path.join(outdir, f'{img_name.replace(".png", "")}_contours.png'))




# CLI
class InferenceCellViTMoNuSegParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference for MoNuSeg dataset",
        )

        parser.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
            default="/homes/fhoerst/histo-projects/CellViT/results/PanNuke/Revision/CellViT/Common-Loss/SAM-H/x20/Fold-1-x20/checkpoints/latest_checkpoint.pth",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            help="Path to MoNuSeg dataset.",
            default="/projects/datashare/tio/histopathology/public-datasets/MoNuSeg/1024/testing",
        )
        parser.add_argument(
            "--outdir",
            type=str,
            help="Path to output directory to store results.",
            default="/homes/fhoerst/histo-projects/CellViT/results/PanNuke/Revision/CellViT/Common-Loss/SAM-H/x20/Fold-1-x20/MoNuSeg/x40/256_64",
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference. Default: 0", default=0
        )
        parser.add_argument(
            "--magnification",
            type=int,
            help="Dataset Magnification. Either 20 or 40. Default: 40",
            choices=[20, 40],
            default=40,
        )
        parser.add_argument(
            "--patching",
            type=bool,
            help="Patch to 256px images. Default: False",
            default=True,
        )
        parser.add_argument(
            "--overlap",
            type=int,
            help="Patch overlap, just valid for patching",
            default=64,
        )
        parser.add_argument(
            "--plots",
            type=bool,
            help="Generate result plots. Default: False",
            default=True,
        )

        # add: allow inference on multiple folders of png patches
        parser.add_argument(
            "--datasets_dir",
            type = str,
            help="This directory should contain PNG patches or folders of PNG patches"
        )

        parser.add_argument(
            "--outputs_dir",
            type = str,
            help="This directory will store predictions folders"

        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":

    import os
    import shutil

    configuration_parser = InferenceCellViTMoNuSegParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)


    ## path/to/png folders
    ## for sam model
    # absolute_output_path = '/home/guoj5/Desktop/200_annotation/augment_model_predictions/cellvit_sam'

    # # update orignal cellvit
    # absolute_input_path = '/home/guoj5/Documents'
    # relative_input_path = '200_annotation_qa'
    # path_to_patch_folder = os.path.join(absolute_input_path, relative_input_path)
    # absolute_output_path = '/home/guoj5/Desktop/200_annotation/temp/cellvit_pred'

    relative_input_path = os.path.basename(configuration['datasets_dir'])
    path_to_patch_folder = configuration['datasets_dir']
    absolute_output_path = configuration['outputs_dir']

    # pre-load the list of directories for inference
    if os.path.exists(relative_input_path + '_dataset_dirs.pkl'):
        with open(relative_input_path + '_dataset_dirs.pkl', 'rb') as file:
            dataset_dirs = pickle.load(file)
    else:
    # save the list of directories as .pkl
        dataset_dirs = list(find_files(path_to_patch_folder, format='.png'))
        with open(relative_input_path + '_dataset_dirs.pkl', 'wb') as binary_file:
            pickle.dump(dataset_dirs, binary_file)

    resume_dataset_dir = 0
    for i in tqdm.tqdm(range(resume_dataset_dir, len(dataset_dirs))):
        print(i)
        dataset = dataset_dirs[i]
        output = os.path.join(absolute_output_path, dataset[dataset.find(relative_input_path):])

        configuration["outdir"] = output
        configuration["dataset"] = dataset

        inf = MoNuSegInference(
        model_path=configuration["model"],
        dataset_path=configuration["dataset"],
        outdir=configuration["outdir"],
        gpu=configuration["gpu"],
        patching=configuration["patching"],
        magnification=configuration["magnification"],
        overlap=configuration["overlap"],)

        inf.run_inference(generate_plots=configuration["plots"])


