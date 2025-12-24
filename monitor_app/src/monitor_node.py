#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import io
import time
import sys
import pickle
import logging
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
project_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'monitor_app'))


from monitor_app.src.monitor import load_model, status_monitor, MonitorFSM, AnormallyFSM

logging.basicConfig(level=logging.INFO)

class MonitorNode(Node):
    def __init__(self):
        super().__init__('monitor_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Bool, 'monitor_warning', 10)
        self.current_frame = None
        # monitor_msg/msg/MonitorMsg.msg
        self.monitor_warning = False
        self.error_description = ""
        self.cur_subtask_idx = -1
        self.cur_prompt = ""
        self.value_function = 0
        self.task_status = 0
        self.reserve1 = False
        self.reserve2 = False
        self.reserve3 = False
        self.reserve4 = False
        self.reserve5 = False
        self.reserve6 = 0
        self.reserve7 = 0
        self.reserve8 = 0
        self.reserve9 = 0
        self.reserve10 = 0
        self.reserve11 = 0.0
        self.reserve12 = 0.0
        self.reserve13 = 0.0
        self.reserve14 = 0.0
        self.reserve15 = 0.0
        self.reserve16 = ""
        self.reserve17 = ""
        self.reserve18 = ""
        self.reserve19 = ""
        self.reserve20 = ""       

    def image_callback(self, msg):
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("Monitor Frame", self.current_frame)
        cv2.waitKey(1)

    def image_issue(self):
        if self.current_frame is None:
            return False
        if np.mean(self.current_frame) < 10:
            print("Warning: Image frame is too dark!")
            return True
        print("Image frame is normal.")
        return False
    
    def publish_msg(self):
        msg = Bool()
        msg.data = self.monitor_warning
        self.publisher_.publish(msg)

    def run(self):
        train_config = json.load(open("data_configs/train_config.json", "r"))
        dino_classifier, data_config = load_model('./checkpoints/dino_classifier.pth', train_config["class_names"])
        with open("./checkpoints/anormally_detect.pkl", 'rb') as file:
            clf = pickle.load(file)

        img_size = (train_config["image_size"][0], train_config["image_size"][1])

        monitor_fsm = MonitorFSM(filter_time=0.1, fps=10)
        anormally_fsm = AnormallyFSM(filter_time=0.1, fps=10)

        while rclpy.ok():
            rclpy.spin_once(self)
            self.monitor_warning = self.image_issue()
            image_cv = self.current_frame
            color_converted_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_path = PILImage.fromarray(color_converted_image)
            status, abnormal, status_candidate, detect, duration, dist = status_monitor(
                image_path, 
                monitor_fsm, 
                anormally_fsm, 
                dino_classifier, 
                data_config, 
                img_size, 
                train_config["class_names"], 
                clf
            )
            logging.info(f"Status: {status}, Abnormal: {abnormal}, Duration: {duration:.2f} sec, Dist: {dist[0]:.2f}")
            self.publish_msg()
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    monitor_node = MonitorNode()
    try:
        monitor_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        monitor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()