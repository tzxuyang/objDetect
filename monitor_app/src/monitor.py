from hmac import new
import sys
import os

import cv2

# Get the absolute path to the directory containing 'src'
# Adjust the path based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
project_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from PIL.ImageShow import show
import timm
import logging
from dataclasses import dataclass
from src.dino_train import DinoClassifier, set_seed
import json
import torch
import pandas as pd
import io
import time
import pickle
from utils import add_text_2_img, record_video_from_images, pad_vit_input, concat_images
from PIL import Image
import numpy as np

# _PROJECT_NAME = "dino_classifier_177_dino_large"

_SEED = 77
_SVM_THRES = -0.55

@dataclass
class ClassifierConfig:
    checkpoint: str = "./checkpoints/dino_classifier.pth" # yolo prediction check point

def load_model(checkpoint, class_names):
    dino_classifier = DinoClassifier(num_classes=len(class_names))
    data_config = timm.data.resolve_model_data_config(dino_classifier.backbone)
    print(data_config)
    # Load checkpoint onto CPU first to avoid CUDA device-deserialization errors, then move to available device
    state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    dino_classifier.load_state_dict(state_dict)
    dino_classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dino_classifier.eval()
    return dino_classifier, data_config

def vit_predict(
    model,
    data_config,
    left_image,
    right_image,
    new_size,
    class_names,
    padding_bbox_left=None,
    padding_bbox_right=None,
):
    if padding_bbox_left is not None and padding_bbox_left != (0, 0, 0, 0):
        print(f"Applying padding to left image with bbox: {padding_bbox_left}")
        left_image = pad_vit_input(left_image, bbox=padding_bbox_left)
    if right_image is not None and padding_bbox_right is not None and padding_bbox_right != (0, 0, 0, 0):
        print(f"Applying padding to right image with bbox: {padding_bbox_right}")
        right_image = pad_vit_input(right_image, bbox=padding_bbox_right)

    image = concat_images(left_image, right_image) if right_image is not None else left_image
    input_tensor = model.process_image(data_config, image, new_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        class_name, confidence, feature = model.predict(input_tensor, return_feature = True, class_names=class_names)
        return class_name, confidence, feature

class AnormallyFSM:
    def __init__(self, filter_time = 0.2, fps = 30):
        # 0 for normal state, 1 for abnormal state
        self.state = 0  # Initial state
        self.state_lst = 0 # Initial state memory
        self.fps = fps
        self.filter_frames = max(1, int(filter_time * fps))
        self.prediction_history = [0] * self.filter_frames # To store recent predictions for filtering

    def get_state_info(self):
        return self.state, self.state_lst, self.prediction_history
    
    def trainsition(self, prediction):
        predict = 1 if prediction[0] == -1 else 0
        self.state_lst = self.state

        if self.state == 0:
            if all(predict == 1 for predict in self.prediction_history) and predict == 1:
                self.state = 1
            else:
                self.state = 0
        else:
            if all(predict == 0 for predict in self.prediction_history) and predict == 0:
                self.state = 0
            else:
                self.state = 1
            
        self.prediction_history.pop(0)
        self.prediction_history.append(predict)

        return self.state
    
class MonitorFSM:
    def __init__(self, filter_time_warn = 0.3, filter_time_recover = 1.0, fps = 30):
        self.state = 0  # Initial state
        self.state_lst = 0 # Initial state memory
        self.fps = fps
        self.filter_frames_warn = max(1, int(filter_time_warn * fps))
        self.filter_frames_recover = max(1, int(filter_time_recover * fps))
        self.prediction_history_warn = [0] * self.filter_frames_warn # To store recent predictions for filtering
        self.prediction_history_recover = [0] * self.filter_frames_recover # To store recent predictions for filtering
        self.timer = 0
        
    def get_state_info(self):
        return self.state, self.state_lst, self.prediction_history_warn, self.prediction_history_recover
    
    def get_state_timer(self):
        return self.timer
    
    def transition(self, prediction, dt=None):
        if dt is None:
            dt = 1 / self.fps
        self._run_timer(dt)
        self.state_lst = self.state
        # Define your state transition logic here
        if self.state == 0:
            if all(predict == 1 for predict in self.prediction_history_warn) and prediction == 1:
                self.state = 1
                self._reset_timer()
            else:
                self.state = 0
        elif self.state == 1:
            if all(predict == 0 for predict in self.prediction_history_recover) and prediction == 0:
                self.state = 0
                self._reset_timer()
            else:
                self.state = 1

        self.prediction_history_warn.pop(0)
        self.prediction_history_warn.append(prediction)
        self.prediction_history_recover.pop(0)
        self.prediction_history_recover.append(prediction)

        return self.state
    
    def _run_timer(self, dt):
        self.timer += dt

    def _reset_timer(self):
        self.timer = 0

class PnpMonitorFSM:
    def __init__(self, filter_time = 0.1, fps = 30):
        self.state = 0  # Initial state
        self.state_lst = 0 # Initial state memory
        self.fps = fps
        self.filter_frames = max(1, int(filter_time * fps))
        self.prediction_history = [0] * self.filter_frames # To store recent predictions for filtering
        self.timer = 0
        
    def get_state_info(self):
        return self.state, self.state_lst, self.prediction_history
    
    def get_state_timer(self):
        return self.timer
    
    def transition(self, prediction, dt=None):
        if dt is None:
            dt = 1 / self.fps
        self._run_timer(dt)
        self.state_lst = self.state
        # Define your state transition logic here
        if self.state == 0:
            if all(predict == 1 for predict in self.prediction_history) and prediction == 1:
                self.state = 1
                self._reset_timer()
            elif all(predict == 2 for predict in self.prediction_history) and prediction == 2:
                self.state = 2
                self._reset_timer()
            else:
                self.state = 0
        elif self.state == 1:
            if all(predict == 0 for predict in self.prediction_history) and prediction == 0:
                self.state = 0
                self._reset_timer()
            else:
                self.state = 1
        elif self.state == 2:
            if all(predict == 0 for predict in self.prediction_history) and prediction == 0:
                self.state = 0
                self._reset_timer()
            else:
                self.state = 2

        self.prediction_history.pop(0)
        self.prediction_history.append(prediction)

        return self.state
    
    def _run_timer(self, dt):
        self.timer += dt

    def _reset_timer(self):
        self.timer = 0

def status_monitor(current_left_frame, current_right_frame, monitor_fsm, anormally_fsm, svm_thres, dino_classifier, data_config, img_size, class_names, padding_bbox_left, padding_bbox_right, clf):
    class_name, confidence, feature = vit_predict(
        dino_classifier,
        data_config,
        current_left_frame,
        current_right_frame,
        img_size,
        class_names,
        padding_bbox_left,
        padding_bbox_right,
    )
    feature = feature.detach().cpu().numpy()
    dist = clf.decision_function(feature)
    detect = [1] if dist > svm_thres else [-1]

    class2int = {name: idx for idx, name in enumerate(class_names)}
    status_candidate = class2int[class_name]
    print(f"status_candidate: {class_name} and confidence: {confidence:.4f}")
    monitor_fsm.transition(status_candidate)
    status = monitor_fsm.state
    duration = monitor_fsm.get_state_timer()    
    anormally_fsm.trainsition(detect)
    abnormal = anormally_fsm.state

    return status, abnormal, status_candidate, detect, duration, dist

if __name__ == "__main__":
    # python monitor_app/src/monitor.py --checkpoint ./checkpoints/dino_classifier.pth --image ./images/port_2.jpg
    train_config = json.load(open("data_configs/train_config_port.json", "r"))
    img_size = (train_config["image_size"][0]*2, train_config["image_size"][1])
    dino_classifier, data_config = load_model('./checkpoints/dino_classifier.pth', train_config["class_names"])
    with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
        clf = pickle.load(file)
    
    set_seed(_SEED)
    # predict(config.checkpoint, config.image, train_config["class_names"])

    df = pd.read_parquet('./videos/port_0002.parquet')
    df['status'] = None
    df['status_filtered'] = None
    df['dist'] = None
    df['abnormal'] = None
    df['abnormal_filtered'] = None
    df['image_new'] = None
    df['duration'] = 0
    count = 0

    monitor_fsm = MonitorFSM(filter_time=0.1, fps=30)
    anormally_fsm = AnormallyFSM(filter_time=0.1, fps=30)

    status = 0
    status_lst = 0

    start_time = time.perf_counter()
    for index, row in df.iterrows():

        logging.info(f"Processing frame {index}")
        image_bytes = row['image']
        image_stream = io.BytesIO(image_bytes)
        image_path = Image.open(image_stream)
        # Convert bytes back to image array if necessary
        # Here we assume the model can take bytes directly; otherwise, convert as needed
        class_name, confidence, feature = vit_predict(
            dino_classifier,
            data_config,
            image_path,
            None,
            img_size,
            train_config["class_names"],
            None,
            None,
        )

        padding_bbox = (train_config["padding_bbox"][0], train_config["padding_bbox"][1], train_config["padding_bbox"][2], train_config["padding_bbox"][3])
        status, abnormal, status_candidate, detect, duration, dist = status_monitor(
            image_path,
            None,
            monitor_fsm, 
            anormally_fsm,
            _SVM_THRES,
            dino_classifier, 
            data_config, 
            img_size, 
            train_config["class_names"], 
            padding_bbox,
            None,
            clf
        )

        int2class = {idx: name for idx, name in enumerate(train_config["class_names"])}
        duration_text = f"{duration:.2f} sec in current state"

        df.loc[index, "status"] = status_candidate
        df.loc[index, "status_filtered"] = status
        df.loc[index, "dist"] = dist
        df.loc[index, "abnormal"] = 1 if detect[0] == -1 else 0
        df.loc[index, "abnormal_filtered"] = abnormal
        image_temp = io.BytesIO(add_text_2_img(image_path, int2class[status]))
        image_path = Image.open(image_temp)
        if duration > 4.5:
            image_temp = io.BytesIO(add_text_2_img(image_path, duration_text, font_size=20, xy = (20, 80)))
            image_path = Image.open(image_temp)
            df.loc[index, "image_new"] = add_text_2_img(image_path, "Warning: too long", font_size=20, xy = (20, 120), color = (255, 0, 0))
        elif abnormal == 1:
            image_temp = io.BytesIO(add_text_2_img(image_path, duration_text, font_size=20, xy = (20, 80)))
            image_path = Image.open(image_temp)
            df.loc[index, "image_new"] = add_text_2_img(image_path, "Warning: abnormal status", font_size=20, xy = (20, 120), color = (255, 0, 0))
        else:
            df.loc[index, "image_new"] = add_text_2_img(image_path, duration_text, font_size=20, xy = (20, 80))
        df.loc[index, "duration"] = duration

        count += 1

        if count % 100 == 0:
            logging.info(f"{count/30} sec classified as {class_name} with confidence {confidence:.4f}")
            image_path.show()

    end_time = time.perf_counter()
    logging.info(f"Processed {count} frames in {end_time - start_time:.2f} seconds.")
    df_new = df[['timestamp_sec', 'status', 'status_filtered', 'dist', 'abnormal', 'abnormal_filtered', 'duration']]
    df_new.to_csv('./videos/port_0002_status.csv', index=False)
    print(df_new.head())

    record_video_from_images(df, 'image_new', fps=30, output_path='./videos/monitor_video_2.mp4')

    