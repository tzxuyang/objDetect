"""
python yolo_detect.py --mode train --dataset ./data_configs/port0.yaml
python yolo_detect.py --mode predict --image ./images/circular_port_22.jpg --checkpoint ./runs/detect/train/weights/best.pt
python yolo_detect.py --mode autolabel --train_image /home/yang/MyRepos/tensorRT/datasets/port0/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port0/labels/train 
--val_image /home/yang/MyRepos/tensorRT/datasets/port0/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port0/labels/val
"""
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
from src.auto_labeling import ObjectDetector, CreateYoloDataset

# auto label model path
_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

@dataclass
class ObjectDetect:
    mode: str # modes with options ["train", "predict", "autolabel"]
    dataset: str = "./data_configs/port0.yaml" # yolo dataset yaml file path
    checkpoint: str = "./yolo11n.pt" # yolo prediction check point
    image: str = "./images/cat1.jpg" # image path
    train_image: str = "/home/yang/MyRepos/tensorRT/datasets/port0/images/train"  # autolabeling train image path
    train_label: str = "/home/yang/MyRepos/tensorRT/datasets/port0/labels/train" # autolabeling train label writing path
    val_image: str = "/home/yang/MyRepos/tensorRT/datasets/port0/images/val" # autolabeling val image path
    val_label: str = "/home/yang/MyRepos/tensorRT/datasets/port0/labels/val" # autolabeling val label writing path


if __name__ == "__main__":
    config = tyro.cli(ObjectDetect)
    if config.mode == "train":
        yolo_port = YOLOCustom("./checkpoints/yolov8n.pt")
        yolo_port.train(data=config.dataset, epochs=100, imgsz=640)
    elif config.mode == "predict":
        yolo_port = YOLOCustom("./checkpoints/yolov8n.pt")
        yolo_port.load_model(config.checkpoint)
        results = yolo_port.predict(img_path=config.image, conf=0.25)

        height, width = results[0].orig_shape
        label_dict = results[0].names
        labels = [label_dict[cls] for cls in results[0].boxes.cls.tolist()]
        bboxs = results[0].boxes.xyxy.tolist()
        draw_bbox(
            config.image,
            bboxs[:],
            labels[:],
            new_size = (width, height)
        )
        plt.show()
    elif config.mode == "autolabel":
        ObjDetectLabeler = ObjectDetector(_MODEL_PATH)
        CreateYoloLabel = CreateYoloDataset({"circular port": 0, "rectangular port": 1})

        # create train data set
        root_dir = config.train_image
        trgt_dir = config.train_label
        path_list = create_file_list(root_dir)

        for path in path_list:
            file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
            logging.info(f"Processing image: {path}")
            result = ObjDetectLabeler.detect_object(path, "circular ports on the white board", max_new_tokens=2048)
            bboxs, labels = ObjDetectLabeler.extract_bbox(result)
            try:
                CreateYoloLabel.create_yolo_label(
                    bboxs, 
                    labels, 
                    (1000, 1000),  # (height, width)
                    file_name
                )
            except:
                logging.info(f"Result is {result} and bboxs are {bboxs} and labels are {labels}")
            draw_bbox(
                path,
                bboxs[:],
                labels[:],
                new_size = (1000, 1000)
            )
            logging.info(f"Saved label file as {file_name}.")

        # create val data set
        root_dir = config.val_image
        trgt_dir = config.val_label
        path_list = create_file_list(root_dir)

        for path in path_list:
            file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
            logging.info(f"Processing image: {path}")
            result = ObjDetectLabeler.detect_object(path, "circular ports on the white board", max_new_tokens=2048)
            bboxs, labels = ObjDetectLabeler.extract_bbox(result)
            try:
                CreateYoloLabel.create_yolo_label(
                    bboxs, 
                    labels, 
                    (1000, 1000),  # (height, width)
                    file_name
                )
            except:
                logging.info(f"Result is {result} and bboxs are {bboxs} and labels are {labels}")
            draw_bbox(
                path,
                bboxs[:],
                labels[:],
                new_size = (1000, 1000)
            )
            logging.info(f"Saved label file as {file_name}.")
        plt.show()
    else:
        logging.info("wrong mode")
