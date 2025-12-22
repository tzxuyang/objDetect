import sys
import os

# Get the absolute path to the directory containing 'src'
# Adjust the path based on your project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
project_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import cv2
import timm
import tyro 
import logging
from dataclasses import dataclass
from src.dino_train import DinoClassifier, set_seed
import json
import torch
import pandas as pd
import numpy as np
import io
import time
import pickle
from utils import add_text_2_img, record_video_from_images
from PIL import Image

# _PROJECT_NAME = "dino_classifier_177_dino_large"
_NUM_CLASSES = 6
_SEED = 77
_CLASS2INT = {"unplugged": 0, "port_1": 1, "port_2": 2, "port_3": 3, "port_4": 4, "port_5": 5}
_INT2CLASS = {0: "unplugged", 1: "port_1", 2: "port_2", 3: "port_3", 4: "port_4", 5: "port_5"}
_SVM_THRES = -0.4

@dataclass
class ClassifierConfig:
    checkpoint: str = "./checkpoints/dino_classifier.pth" # yolo prediction check point

def load_model(checkpoint, class_names):
    dino_classifier = DinoClassifier(num_classes=len(class_names))
    data_config = timm.data.resolve_model_data_config(dino_classifier.backbone)
    dino_classifier.load_state_dict(torch.load(checkpoint))
    dino_classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dino_classifier.eval()
    return dino_classifier, data_config

def vit_predict(model, data_config, image_path, new_size, class_names):
    input_tensor = model.process_image(data_config, image_path, new_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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
            print(dt)
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
            elif all(predict == 3 for predict in self.prediction_history) and prediction == 3:
                self.state = 3
                self._reset_timer() 
            elif all(predict == 4 for predict in self.prediction_history) and prediction == 4:
                self.state = 4
                self._reset_timer()
            elif all(predict == 5 for predict in self.prediction_history) and prediction == 5:
                self.state = 5
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
        elif self.state == 3:
            if all(predict == 0 for predict in self.prediction_history) and prediction == 0:
                self.state = 0
                self._reset_timer()
            else:
                self.state = 3
        elif self.state == 4:
            if all(predict == 0 for predict in self.prediction_history) and prediction == 0:
                self.state = 0
                self._reset_timer()
            else:
                self.state = 4
        elif self.state == 5: 
            if all(predict == 0 for predict in self.prediction_history) and prediction == 0:
                self.state = 0
                self._reset_timer()
            else:
                self.state = 5

        self.prediction_history.pop(0)
        self.prediction_history.append(prediction)

        return self.state
    
    def _run_timer(self, dt):
        self.timer += dt

    def _reset_timer(self):
        self.timer = 0


if __name__ == "__main__":
    # python monitor_src/monitor.py --checkpoint ./checkpoints/dino_classifier.pth --image ./images/port_2.jpg
    train_config = json.load(open("data_configs/train_config.json", "r"))
    dino_classifier, data_config = load_model('./checkpoints/dino_classifier.pth', train_config["class_names"])
    with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
        clf = pickle.load(file)
    
    img_size = (train_config["image_size"][0], train_config["image_size"][1])
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
        class_name, confidence, feature = vit_predict(dino_classifier, data_config, image_path, img_size, train_config["class_names"])


        feature = feature.detach().cpu().numpy()
        # detect = clf.predict(feature)
        dist = clf.decision_function(feature)
        detect = [1] if dist > _SVM_THRES else [-1]


        status_candidate = _CLASS2INT[class_name]

        monitor_fsm.transition(status_candidate)
        status = monitor_fsm.state
        status_lst = monitor_fsm.state_lst
        duration = monitor_fsm.get_state_timer()

        anormally_fsm.trainsition(detect)
        abnormal = anormally_fsm.state
     
        status_text = _INT2CLASS[status]
        duration_text = f"{duration:.2f} sec in current state"

        df.loc[index, "status"] = status_candidate
        df.loc[index, "status_filtered"] = status
        df.loc[index, "dist"] = dist
        df.loc[index, "abnormal"] = 1 if detect[0] == -1 else 0
        df.loc[index, "abnormal_filtered"] = abnormal
        image_temp = io.BytesIO(add_text_2_img(image_path, status_text))
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

    