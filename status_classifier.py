import sys
import os
from cv2 import data
import timm
from torch.quantization import default_qat_qconfig
# Get the absolute path to the directory containing 'src'
# Adjust the path based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

from timm.models import checkpoint
import tyro 
import logging
from PIL import Image
from dataclasses import dataclass, field, fields, MISSING
from src.auto_labeling import classifier_autolabel_complex, create_video, batch_label, PassFailDataset
from src.dino_train import DinoClassifier, set_seed, train_classifier
from src.utils import pad_vit_input, concat_images
import json
import pickle
import torch

_PROJECT_NAME = "dino_classifier_177_dinov3_small"
# _PROJECT_NAME = "dino_classifier_177_dino_large"
_WANDB_KEY = "93205eda06a813b688c0462d11f09886a0cf7ae8"
_SEED = 77

@dataclass
class ClassifierConfig:
    mode: str # modes with options ["train", "predict", "autolabel"]
    project_name: str = _PROJECT_NAME # wandb project name
    wandb_key: str = _WANDB_KEY # wandb api key
    checkpoint: str = "./checkpoints/dino_classifier.pth" # yolo prediction check point
    image_left: str = "default_value" # left image path for predict mode
    image_right: str = "default_value" # right image path for predict mode
    label_config: str = "data_configs/label_config_ioboard.json" # data config json file path
    train_config: str = "data_configs/train_config_port.json" # training config json file path
    train_image: str = "default_value"  # autolabeling train image path
    train_label: str = "default_value" # autolabeling train label writing path
    val_image: str = "default_value" # autolabeling val image path
    val_label: str = "default_value" # autolabeling val label writing path
    image_list: list[str] = field(default_factory=list)  # autolabeling raw image list path

def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))

def _infer_output_root(train_image, train_label, val_image, val_label):
    resolved_paths = [
        _resolve_repo_path(train_image),
        _resolve_repo_path(train_label),
        _resolve_repo_path(val_image),
        _resolve_repo_path(val_label),
    ]
    output_root = os.path.commonpath(resolved_paths)
    return output_root

def predict(checkpoint, image_path, new_size, class_names, data_config=None):
    dino_classifier = DinoClassifier(num_classes=len(class_names))
    # Load checkpoint onto CPU first to avoid CUDA device-deserialization errors, then move model to device
    try:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint}: {e}")
    dino_classifier.load_state_dict(state_dict)
    dino_classifier.to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dino_classifier.eval()

    if data_config is None:
        data_config = timm.data.resolve_model_data_config(dino_classifier.backbone)
    input_tensor = dino_classifier.process_image(data_config, image_path, new_size).to(device)

    with torch.no_grad():
        class_name, confidence, feature = dino_classifier.predict(input_tensor, return_feature = True, class_names=class_names)

    return class_name, confidence, feature

def anormally_detect(model, feature_array):
    return model.predict(feature_array)


if __name__ == "__main__":
    config = tyro.cli(ClassifierConfig)
    # uv run status_classifier.py --mode autolabel_label --label_config data_configs/label_config_ioboard.json
    if config.mode == "autolabel":
        label_config = json.load(open(config.label_config, "r"))
        output_root = _infer_output_root(
            label_config["train_image"],
            label_config["train_label"],
            label_config["val_image"],
            label_config["val_label"],
        )

        dataset = PassFailDataset(
            success_path_left=label_config["success_path_left"],
            success_path_right=label_config["success_path_right"],
            fail_path_left=label_config["fail_path_left"],
            fail_path_right=label_config["fail_path_right"],
            fail_fps=label_config.get("fail_fps", 60),
            fail_duration=label_config.get("fail_duration", 0.5),
            success_fps=label_config.get("success_fps", 1),
            left_mask_bbox=label_config["padding_bbox_left"],
            right_mask_bbox=label_config["padding_bbox_right"],
            val_ratio=label_config.get("val_ratio", 0.2),
            output_path=output_root,
        )
        dataset.create_train_val_split()
        logging.info(
            "Validation success files: %s",
            [os.path.basename(path) for path in dataset.val_success_files_left],
        )
        logging.info(
            "Validation fail files: %s",
            [os.path.basename(path) for path in dataset.val_fail_files_left],
        )
        dataset.create_train_classification_dataset()
        dataset.create_val_classification_dataset()
        logging.info("Created classification dataset under %s", output_root)
        
    elif config.mode == "train":
    # uv run status_classifier.py --mode train --train_config data_configs/train_config_pnp.json --project_name dino_classifier_177_dinov3_small
        train_config = json.load(open(config.train_config, "r"))
        img_size = (train_config["image_size"][0]*2, train_config["image_size"][1])
        padding_bbox = (train_config["padding_bbox"][0], train_config["padding_bbox"][1], train_config["padding_bbox"][2], train_config["padding_bbox"][3])
        train_classifier(
            project_name=config.project_name,
            train_file_directory=train_config["train_image"],
            train_label_directory=train_config["train_label"],
            test_file_directory=train_config["val_image"],
            test_label_directory=train_config["val_label"],
            train_cluster=True,
            new_size=img_size,
            padding_bbox=padding_bbox,
            class_names=train_config["class_names"],
            class_weights=train_config["class_weights"],
            batch_size=train_config["batch_size"],
            lr_max=train_config["lr_max"],
            lr_min=train_config["lr_min"],
            epoch=train_config["epoch"],
        )

    else:
    # uv run status_classifier.py --mode predict --train_config data_configs/train_config_ioboard.json --checkpoint ./checkpoints/dino_classifier.pth --image-left ./images/left.jpg --image-right ./images/right.jpg
        train_config = json.load(open(config.train_config, "r"))
        set_seed(_SEED)
        with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
            clf = pickle.load(file)
        img_size = (train_config["image_size"][0]*2, train_config["image_size"][1])
        if config.image_left == "default_value" or config.image_right == "default_value":
            raise ValueError("predict mode requires both --image-left and --image-right")

        padding_bbox_left = tuple(train_config.get("padding_bbox_left", train_config.get("padding_bbox", [0, 0, 0, 0])))
        padding_bbox_right = tuple(train_config.get("padding_bbox_right", train_config.get("padding_bbox", [0, 0, 0, 0])))

        left_image = Image.open(_resolve_repo_path(config.image_left)).convert("RGB")
        right_image = Image.open(_resolve_repo_path(config.image_right)).convert("RGB")

        if padding_bbox_left != (0, 0, 0, 0):
            left_image = pad_vit_input(left_image, bbox=padding_bbox_left)
        if padding_bbox_right != (0, 0, 0, 0):
            right_image = pad_vit_input(right_image, bbox=padding_bbox_right)

        image = concat_images(left_image, right_image)
        image.show()
        class_name, confidence, feature = predict(config.checkpoint, image, img_size, train_config["class_names"])
        logging.info(
            "%s + %s classified as %s with confidence %.4f",
            config.image_left,
            config.image_right,
            class_name,
            confidence,
        )
        feature = feature.detach().cpu().numpy()
        detect = clf.predict(feature)
        logging.info(f"anormally result detection {detect}")
    