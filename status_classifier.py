import sys
import os

# Get the absolute path to the directory containing 'src'
# Adjust the path based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

import tyro 
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.utils import draw_bbox, create_file_list
from src.yolo_train import YOLOCustom
from src.auto_labeling import AiLabeler, CreateYoloDataset, CreateClassDataset, classification_autolabel

# auto label model path
_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

if __name__ == "__main__":
    label_prompt = """
    Please describe this image. Descrbe if the cable with green port is plugged in to the circular port and which port it is connected to in the format with as follows:
    {
        "description": "",
        "Is plugged in": true/false, 
        "connected to port (int)": "",
    }
    """
    mode = "autolabel"

    if mode == "autolabel":
        classifier = AiLabeler(_MODEL_PATH)
        ClassLabel = CreateClassDataset()

        # create train data set
        root_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/train"
        trgt_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train"
        path_list = create_file_list(root_dir)

        for i, path in enumerate(path_list):
            file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
            logging.info(f"Processing image: {path}")
            result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, file_name, label_prompt, max_new_tokens=1024)
            if i % 5 == 0 and i < 50:
                logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")
            logging.info(f"Saved label file as {file_name}.")

        # create val data set
        root_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
        trgt_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val"
        path_list = create_file_list(root_dir)

        for i, path in enumerate(path_list):
            file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
            logging.info(f"Processing image: {path}")
            result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, file_name, label_prompt, max_new_tokens=1024)
            if i % 5 == 0 and i < 50:
                logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")
            logging.info(f"Saved label file as {file_name}.")


    