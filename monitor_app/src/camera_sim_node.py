#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import subprocess

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hw_decoders_any;none")

import cv2
import numpy as np
import logging
import json
import time
import tyro
import sys
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

_TIMER_PERIOD = 0.0333  # Approx 30 FPS
_SNAPSHOT_INTERVAL = 1.0

@dataclass
class Config:
    config_path: str
    save_image: bool = False

def read_config(config_path):
    monitor_config = json.load(open(config_path, "r"))
    video_path_left = monitor_config.get("video_path_left")
    video_path_right = monitor_config.get("video_path_right")
    img_size = monitor_config.get("image_size")
    return video_path_left, video_path_right, (img_size[0], img_size[1])

def get_video_size(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        video_path,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    width_text, height_text = result.stdout.strip().split("x")
    return int(width_text), int(height_text)

class FFmpegVideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.width, self.height = get_video_size(video_path)
        self.frame_size = self.width * self.height * 3
        self.process = None
        self._start()

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-i",
            self.video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read(self):
        raw_frame = self.process.stdout.read(self.frame_size)
        if len(raw_frame) != self.frame_size:
            self._start()
            raw_frame = self.process.stdout.read(self.frame_size)
            if len(raw_frame) != self.frame_size:
                return False, None

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame.copy()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

class ImagePublisher(Node):
    def __init__(self, img_size, video_path_left, video_path_right, save_image=False):
        super().__init__('image_publisher')
        self.left_publisher_ = self.create_publisher(Image, '/camera/camera/color/image_rect_left', 10)
        self.right_publisher_ = self.create_publisher(Image, '/camera/camera/color/image_rect_right', 10)
        self.video_path_left = video_path_left
        self.video_path_right = video_path_right
        self.save_image = save_image
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.snapshot_dir = os.path.join(repo_root, "images")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.last_snapshot_time = 0.0
        self.snapshot_idx = 0
        timer_period = _TIMER_PERIOD
        self.timer = self.create_timer(timer_period, self.timer_callback)  # Approx 30 FPS
        self.bridge = CvBridge()
        self.left_reader = FFmpegVideoReader(video_path_left)
        self.right_reader = FFmpegVideoReader(video_path_right)
        self.left_cv_image = np.zeros((self.left_reader.height, self.left_reader.width, 3), dtype=np.uint8)
        self.right_cv_image = np.zeros((self.right_reader.height, self.right_reader.width, 3), dtype=np.uint8)
    
    def get_image(self):
        left_ret, left_frame = self.left_reader.read()
        right_ret, right_frame = self.right_reader.read()
        if left_ret and left_frame is not None:
            self.left_cv_image = left_frame
        if right_ret and right_frame is not None:
            self.right_cv_image = right_frame
    
    def pulish_msg(self):
        self.left_publisher_.publish(self.bridge.cv2_to_imgmsg(np.array(self.left_cv_image), "bgr8"))
        self.right_publisher_.publish(self.bridge.cv2_to_imgmsg(np.array(self.right_cv_image), "bgr8"))
        self.get_logger().info('Publishing left and right images')

    def save_snapshot(self):
        current_time = time.monotonic()
        if current_time - self.last_snapshot_time < _SNAPSHOT_INTERVAL:
            return

        left_path = os.path.join(self.snapshot_dir, f"camera_left_{self.snapshot_idx:06d}.jpg")
        right_path = os.path.join(self.snapshot_dir, f"camera_right_{self.snapshot_idx:06d}.jpg")
        cv2.imwrite(left_path, self.left_cv_image)
        cv2.imwrite(right_path, self.right_cv_image)
        self.last_snapshot_time = current_time
        self.snapshot_idx += 1

    def timer_callback(self):
        self.get_image()
        cv2.imshow("Camera Left", self.left_cv_image)
        cv2.imshow("Camera Right", self.right_cv_image)
        cv2.waitKey(1)
        if self.save_image:
            self.save_snapshot()
        self.pulish_msg()

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self)
    
def main(cfg: Config)-> None:
    _VIDEO_PATH_LEFT, _VIDEO_PATH_RIGHT, (_CAM_WIDTH, _CAM_HEIGHT) = read_config(cfg.config_path)
    rclpy.init(args=None)
    image_publisher = ImagePublisher(
        img_size=(_CAM_WIDTH, _CAM_HEIGHT),
        video_path_left=_VIDEO_PATH_LEFT,
        video_path_right=_VIDEO_PATH_RIGHT,
        save_image=cfg.save_image,
    )
    logging.info("Starting image publisher...")
    try:
        image_publisher.run()
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.left_reader.close()
        image_publisher.right_reader.close()
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
   # python monitor_app/src/camera_sim_node.py --config-path data_configs/monitor_config_port.json
   # python monitor_app/src/camera_sim_node.py --config-path data_configs/monitor_config_pnp.json
   normalized_argv = [
       arg.replace("_", "-") if arg.startswith("--") else arg
       for arg in sys.argv[1:]
   ]
   main(tyro.cli(Config, args=normalized_argv))