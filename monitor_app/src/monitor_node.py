#!/usr/bin/env python3
from math import e
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from cv_bridge import CvBridge
from arbitrator_msg.msg import MonitorState
import cv2
import numpy as np
import time
import sys
import pickle
import logging
import os
import json
import tyro
from dataclasses import dataclass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
project_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'monitor_app'))

from monitor_app.src.monitor import load_model, status_monitor, MonitorFSM, AnormallyFSM, PnpMonitorFSM

logging.basicConfig(level=logging.INFO)

_BLACK_THRESHOLD = 10
_FPS = 30
_DOWN_SAMPLE_RATE = 3
_FILTER_TIME = 0.15

@dataclass
class Config:
    config_path: str

def read_config(config_path):
    monitor_config = json.load(open(config_path, "r"))
    duration_threshold = monitor_config.get("duration_thres")
    class_names = monitor_config.get("class_names")
    fsm = monitor_config.get("fsm")
    fsm_thres = monitor_config.get("svm_thres")
    img_size = monitor_config.get("image_size")
    return duration_threshold, class_names, fsm, img_size, fsm_thres

class MonitorNode(Node):
    def __init__(self, duration_threshold, class_names, fsm, img_size, fsm_thres):
        super().__init__('monitor_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10)
        self.count = 0
        self.bridge = CvBridge()
        self.monitor_publisher_ = self.create_publisher(MonitorState, '/monitor/monitor_state', 10)
        self.duration_threshold = duration_threshold
        self.img_size = img_size
        self.class_names = class_names
        self.int2class = {idx: name for idx, name in enumerate(class_names)}
        self.fsm = fsm
        self.fsm_thres = fsm_thres
        self.current_frame = None
        self.monitor_warning = False
        self.error_description = ""
        self.cur_subtask_idx = 0
        self.cur_prompt = ""
        self.value_function = 0
        self.task_status = 0
        # self.reserve1-5 = False
        # self.reserve6-10 = 0
        self.reserve11 = 0.0
        # self.reserve12-15 = 0.0
        # self.reserve16-20 = ""
    

    def _image_edit(self):
        # add_text_2_img(img, text, font_size=40, xy=(20, 20), color=(0, 0, 255)):
        img = self.current_frame

        # 2. Define text parameters
        state_text = self.int2class[self.cur_subtask_idx]
        duration_text = f"{self.reserve11:.2f} sec in current state"
        warning_text = "WARNING!" if self.monitor_warning else ""
        if self.monitor_warning and self.error_description != "":
            warning_text += f" ({self.error_description})"

        state_position = (20, 20) # Bottom-left corner of the text
        duration_position = (20, 40) # Bottom-left corner of the text
        warning_position = (20, 60) # Bottom-left corner of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        state_color = (255, 0, 0) # Blue color in BGR format
        duration_color = (255, 0, 0) # Blue color in BGR format
        warning_color = (0, 0, 255) # Red color in BGR format
        thickness = 2
        line_type = cv2.LINE_AA

        # 3. Add the text to the image using cv2.putText()
        cv2.putText(img, state_text, state_position, font, font_scale, state_color, thickness, line_type)
        cv2.putText(img, duration_text, duration_position, font, font_scale, duration_color, thickness, line_type)
        if self.monitor_warning:
            cv2.putText(img, warning_text, warning_position, font, font_scale, warning_color, thickness, line_type)
        return img

    def image_callback(self, msg):
        if self.count % _DOWN_SAMPLE_RATE == 0:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            edited_frame = self._image_edit()
            cv2.imshow("Monitor Frame", edited_frame)
            cv2.waitKey(1)
        self.count += 1

    def image_issue(self):
        if self.current_frame is None:
            return False
        if np.mean(self.current_frame) < _BLACK_THRESHOLD:
            return True
        return False
    
    def publish_msg(self):
        monitor_state_msg = MonitorState()
        monitor_state_msg.warning = self.monitor_warning
        monitor_state_msg.error_description = self.error_description
        monitor_state_msg.cur_subtask_idx = self.cur_subtask_idx
        self.monitor_publisher_.publish(monitor_state_msg)
        self.get_logger().info(f"Published monitor warning: {self.monitor_warning}, state idx: {self.cur_subtask_idx}")

    def run(self):
        dino_classifier, data_config = load_model('./checkpoints/dino_classifier.pth', self.class_names)
        with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
            clf = pickle.load(file)

        if self.fsm == "MonitorFSM":
            monitor_fsm = MonitorFSM(filter_time=_FILTER_TIME, fps=_FPS)
        elif self.fsm == "PnpMonitorFSM":
            monitor_fsm = PnpMonitorFSM(filter_time=_FILTER_TIME, fps=_FPS)
        anormally_fsm = AnormallyFSM(filter_time=0.2, fps=_FPS)

        while rclpy.ok():
            rclpy.spin_once(self)
            raw_image_issue = self.image_issue()
            image_cv = self.current_frame
            color_converted_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_path = PILImage.fromarray(color_converted_image)
            status, abnormal, _, _, duration, dist = status_monitor(
                image_path, 
                monitor_fsm, 
                anormally_fsm,
                self.fsm_thres, 
                dino_classifier, 
                data_config, 
                self.img_size, 
                self.class_names, 
                clf
            )

            self.cur_subtask_idx = status
            self.reserve11 = duration
            if raw_image_issue or abnormal or duration > self.duration_threshold or status == 2:
                self.monitor_warning = True
                if duration > self.duration_threshold:
                    self.error_description = "Duration Issue"
            else:
                self.monitor_warning = False
        
            logging.info(f"raw image issue: {raw_image_issue}, abnormal status: {abnormal}, dist: {dist} duration in state: {duration:.2f} sec")

            self.publish_msg()
            time.sleep(0.01)

def main(cfg: Config)-> None:
    duration_thres, class_names, fsm, img_size, fsm_thres = read_config(cfg.config_path)
    rclpy.init(args=None)
    monitor_node = MonitorNode(duration_thres, class_names, fsm, img_size, fsm_thres)
    try:
        monitor_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        monitor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # python monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_port.json
    # python monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_pnp.json
    main(tyro.cli(Config))