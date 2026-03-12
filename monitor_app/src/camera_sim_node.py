#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import logging
import json
import tyro
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

_TIMER_PERIOD = 0.0333  # Approx 30 FPS

@dataclass
class Config:
    config_path: str

def read_config(config_path):
    monitor_config = json.load(open(config_path, "r"))
    video_path = monitor_config.get("video_path")
    img_size = monitor_config.get("image_size")
    return video_path, (img_size[0], img_size[1])

class ImagePublisher(Node):
    def __init__(self, img_size, video_path):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/camera/color/image_rect_raw', 10)
        self.video_path = video_path # Replace with your video file
        width, height = img_size
        self.width = width
        self.height = height
        timer_period = _TIMER_PERIOD
        self.timer = self.create_timer(timer_period, self.timer_callback)  # Approx 30 FPS
        self.cv_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(video_path)
    
    def get_image(self):
        ret, frame = self.cap.read()
        if frame is not None:
            self.cv_image = cv2.resize(frame, (self.width, self.height))
    
    def pulish_msg(self):
        self.publisher_.publish(self.bridge.cv2_to_imgmsg(np.array(self.cv_image), "bgr8"))
        self.get_logger().info('Publishing an image')

    def timer_callback(self):
        self.get_image()
        cv2.imshow("Camera", self.cv_image)
        cv2.waitKey(1)
        self.pulish_msg()

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self)
    
def main(cfg: Config)-> None:
    _VIDEO_PATH, (_CAM_WIDTH, _CAM_HEIGHT) = read_config(cfg.config_path)
    rclpy.init(args=None)
    image_publisher = ImagePublisher(img_size = (_CAM_WIDTH, _CAM_HEIGHT), video_path = _VIDEO_PATH)
    logging.info("Starting image publisher...")
    try:
        image_publisher.run()
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
   # python monitor_app/src/camera_sim_node.py --config-path data_configs/monitor_config_port.json
   # python monitor_app/src/camera_sim_node.py --config-path data_configs/monitor_config_pnp.json
   main(tyro.cli(Config))