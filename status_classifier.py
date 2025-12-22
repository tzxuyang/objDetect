import sys
import os
import timm
# Get the absolute path to the directory containing 'src'
# Adjust the path based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

import tyro 
import logging
from dataclasses import dataclass
from src.utils import create_file_list
from src.auto_labeling import AiLabeler, CreateClassDataset, classification_autolabel
from src.dino_train import DinoClassifier, set_seed, train_classifier
import json
import pickle
import torch

# auto label model path
_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
_AUTO_LABEL_SHOW_ITER = 5
_AUTO_LABEL_SHOW_MAX = 50
_PROJECT_NAME = "dino_classifier_177_dinov3_small"
# _PROJECT_NAME = "dino_classifier_177_dino_large"
_WANDB_KEY = "93205eda06a813b688c0462d11f09886a0cf7ae8"
_NUM_CLASSES = 6
_SEED = 77

@dataclass
class ClassifierConfig:
    mode: str # modes with options ["train", "predict", "autolabel"]
    project_name: str = _PROJECT_NAME # wandb project name
    wandb_key: str = _WANDB_KEY # wandb api key
    checkpoint: str = "./checkpoints/dino_classifier.pth" # yolo prediction check point
    image: str = "./images/port_2.jpg" # image path
    train_image: str = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/train"  # autolabeling train image path
    train_label: str = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train" # autolabeling train label writing path
    val_image: str = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val" # autolabeling val image path
    val_label: str = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val" # autolabeling val label writing path

def predict(checkpoint, image_path, new_size, class_names, data_config=None):
    dino_classifier = DinoClassifier(num_classes=_NUM_CLASSES)
    dino_classifier.load_state_dict(torch.load(checkpoint))
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
    
def classifier_autolabel(train_image_dir, train_label_dir, val_image_dir, val_label_dir, label_prompt, max_new_tokens=1024):
    classifier = AiLabeler(_MODEL_PATH)
    ClassLabel = CreateClassDataset()

    # create train data set
    root_dir = train_image_dir
    trgt_dir = train_label_dir
    path_list = create_file_list(root_dir)

    for i, path in enumerate(path_list):
        file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
        logging.info(f"Processing image: {path}")
        result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, file_name, label_prompt, max_new_tokens=max_new_tokens)
        if i % _AUTO_LABEL_SHOW_ITER == 0 and i < _AUTO_LABEL_SHOW_MAX:
            logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")
        logging.info(f"Saved label file as {file_name}.")

    # create val data set
    root_dir = val_image_dir
    trgt_dir = val_label_dir
    path_list = create_file_list(root_dir)

    for i, path in enumerate(path_list):
        file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
        logging.info(f"Processing image: {path}")
        result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, file_name, label_prompt, max_new_tokens=max_new_tokens)
        if i % _AUTO_LABEL_SHOW_ITER == 0 and i < _AUTO_LABEL_SHOW_MAX:
            logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")
        logging.info(f"Saved label file as {file_name}.")

if __name__ == "__main__":
    config = tyro.cli(ClassifierConfig)

    # python status_classifier.py --mode autolabel --train_image /home/yang/MyRepos/tensorRT/datasets/port_cls/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train2 \
    # --val_image /home/yang/MyRepos/tensorRT/datasets/port_cls/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val2
    if config.mode == "autolabel":
        label_prompt = """
        Please describe this image. Descrbe if the cable with green port is plugged in to the circular port and which port it is connected to in the format with as follows:
        {
            "description": "",
            "Is plugged in": true/false, 
            "connected to port (int)": "",
        }
        """
        classifier_autolabel(
            train_image_dir = config.train_image, 
            train_label_dir = config.train_label, 
            val_image_dir = config.val_image, 
            val_label_dir = config.val_label, 
            label_prompt = label_prompt, 
            max_new_tokens=1024
        )

    elif config.mode == "train":
    # python status_classifier.py --mode train --project_name dino_classifier_177_dinov3_small
        train_config = json.load(open("data_configs/train_config.json", "r"))
        img_size = (train_config["image_size"][0], train_config["image_size"][1])
        train_classifier(
            project_name=config.project_name,
            train_file_directory=train_config["train_image"],
            train_label_directory=train_config["train_label"],
            test_file_directory=train_config["val_image"],
            test_label_directory=train_config["val_label"],
            train_cluster=True,
            new_size=img_size,
            class_names=train_config["class_names"],
            batch_size=train_config["batch_size"],
            lr_max=train_config["lr_max"],
            lr_min=train_config["lr_min"],
            epoch=train_config["epoch"],
        )

    else:
    # python status_classifier.py --mode predict --checkpoint ./checkpoints/dino_classifier.pth --image ./images/port_2.jpg
        train_config = json.load(open("data_configs/train_config.json", "r"))
        set_seed(_SEED)
        with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
            clf = pickle.load(file)
        img_size = (train_config["image_size"][0], train_config["image_size"][1])
        class_name, confidence, feature = predict(config.checkpoint, config.image, img_size, train_config["class_names"])
        logging.info(f"{config.image} classified as {class_name} with confidence {confidence:.4f}")
        feature = feature.detach().cpu().numpy()
        detect = clf.predict(feature)
        logging.info(f"anormally result detection {detect}")
    