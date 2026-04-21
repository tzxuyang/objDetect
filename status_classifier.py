import sys
import os
import timm

# Get the absolute path to the directory containing 'src'
# Adjust the path based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

import tyro 
import logging
from PIL import Image
from dataclasses import dataclass, field, fields, MISSING
from src.auto_labeling import classifier_autolabel_complex, classifier_autolabel, create_video, batch_label
from src.dino_train import DinoClassifier, set_seed, train_classifier
from src.utils import pad_vit_input
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
    image: str = "./images/port_2.jpg" # image path
    train_config: str = "data_configs/train_config_port.json" # training config json file path
    train_image: str = "default_value"  # autolabeling train image path
    train_label: str = "default_value" # autolabeling train label writing path
    val_image: str = "default_value" # autolabeling val image path
    val_label: str = "default_value" # autolabeling val label writing path
    image_list: list[str] = field(default_factory=list)  # autolabeling raw image list path

def predict(checkpoint, image_path, new_size, class_names, data_config=None):
    dino_classifier = DinoClassifier(num_classes=len(class_names))
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


if __name__ == "__main__":
    config = tyro.cli(ClassifierConfig)

    # python status_classifier.py --mode autolabel --train_image /home/yang/MyRepos/tensorRT/datasets/port_cls/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train2 \
    # --val_image /home/yang/MyRepos/tensorRT/datasets/port_cls/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val2
    if config.mode == "autolabel":
        label_prompt = """
        Please describe this image. Describe if the cable with a green head is plugged in a circular port, and which port it is in in the format with as follows:
        {
            "description": "",
            "Is plugged in": true/false, 
            "inserted to port (int)": "",
        }
        """
        classifier_autolabel_complex(
            train_image_dir = config.train_image, 
            train_label_dir = config.train_label, 
            val_image_dir = config.val_image, 
            val_label_dir = config.val_label, 
            label_prompt = label_prompt,
            step = 5,
            max_new_tokens=1024
        )
    # # create train data set
    # trgt_dir_img = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/images/train"
    # trgt_dir_label = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/labels/train"
    # root_dir = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/episode0"
    # root_label = "/home/yang/MyRepos/tensorRT/datasets/port_actibot/episode0.txt"
    # batch_label(root_dir, root_label, trgt_dir_img, trgt_dir_label, step=10)

    # python status_classifier.py --mode semi_autolabel_img2vid --image_list /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode0 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode1 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode2 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode3 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode5
    elif config.mode == "semi_autolabel_img2vid":
        image_list = config.image_list
        for image_dir in image_list:
            logging.info(f"Processing image dir: {image_dir} to video")
            trgt_video_file = image_dir + ".mp4"
            create_video(image_dir, trgt_video_file, fps=30)

    # python status_classifier.py --mode semi_autolabel_label --train_image /home/yang/MyRepos/tensorRT/datasets/port_actibot/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port_actibot/labels/train --image_list /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode0 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode1 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode2 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode3 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4
    # python status_classifier.py --mode semi_autolabel_label --val_image /home/yang/MyRepos/tensorRT/datasets/port_actibot/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port_actibot/labels/val --image_list /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode4 /home/yang/MyRepos/tensorRT/datasets/port_actibot/episode5
    elif config.mode == "semi_autolabel_label":
        explicitly_passed = [f.name for f in fields(ClassifierConfig) if getattr(config, f.name) is not MISSING and getattr(config, f.name) != "default_value"]
        if 'train_image' in explicitly_passed and 'train_label' in explicitly_passed:
            trgt_dir_img = config.train_image
            trgt_dir_label = config.train_label
            for image_dir in config.image_list:
                logging.info(f"Processing image dir: {image_dir} to labels in {trgt_dir_label}")
                batch_label(
                    image_dir, 
                    image_dir + ".txt", 
                    trgt_dir_img, 
                    trgt_dir_label,
                    step = 10
                )
        elif 'val_image' in explicitly_passed and 'val_label' in explicitly_passed:
            trgt_dir_img = config.val_image
            trgt_dir_label = config.val_label
            for image_dir in config.image_list:
                logging.info(f"Processing image dir: {image_dir} to labels in {trgt_dir_label}")
                batch_label(
                    image_dir, 
                    image_dir + ".txt", 
                    trgt_dir_img, 
                    trgt_dir_label,
                    step = 20,
                    init_idx = 5
                )
        else:
            logging.error("For semi_autolabel_label mode, please provide either train_image and train_label or val_image and val_label paths.")
        
    elif config.mode == "train":
    # python status_classifier.py --mode train --train_config data_configs/train_config_pnp.json --project_name dino_classifier_177_dinov3_small
        train_config = json.load(open(config.train_config, "r"))
        padding_box = (train_config["padding_bbox"][0], train_config["padding_bbox"][1], train_config["padding_bbox"][2], train_config["padding_bbox"][3])
        img_size = (train_config["image_size"][0], train_config["image_size"][1])
        train_classifier(
            project_name=config.project_name,
            train_file_directory=train_config["train_image"],
            train_label_directory=train_config["train_label"],
            test_file_directory=train_config["val_image"],
            test_label_directory=train_config["val_label"],
            train_cluster=True,
            new_size=img_size,
            padding_bbox=padding_box,
            class_names=train_config["class_names"],
            batch_size=train_config["batch_size"],
            lr_max=train_config["lr_max"],
            lr_min=train_config["lr_min"],
            epoch=train_config["epoch"],
        )

    else:
    # python status_classifier.py --mode predict --train_config data_configs/train_config_pnp.json --checkpoint ./checkpoints/dino_classifier.pth --image ./images/port_2.jpg
        train_config = json.load(open(config.train_config, "r"))
        set_seed(_SEED)
        with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
            clf = pickle.load(file)
        img_size = (train_config["image_size"][0], train_config["image_size"][1])
        image = Image.open(config.image).convert("RGB")
        image = pad_vit_input(image, bbox=[int(img_size[0]*0.2), int(img_size[1]/2), int(img_size[0]*0.8), img_size[1]], target_size=img_size)
        image.show()
        class_name, confidence, feature = predict(config.checkpoint, config.image, img_size, train_config["class_names"])
        logging.info(f"{config.image} classified as {class_name} with confidence {confidence:.4f}")
        feature = feature.detach().cpu().numpy()
        detect = clf.predict(feature)
        logging.info(f"anormally result detection {detect}")
    