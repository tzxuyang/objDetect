import os
import random
import shutil
import subprocess
import numpy as np
import io
from PIL import Image
import torch
import json
import logging
from retrying import retry
import matplotlib.pyplot as plt
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import utils
from utils import create_file_list
import pandas as pd

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hw_decoders_any;none")

import cv2
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO)

# auto label model path
_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
_AUTO_LABEL_SHOW_ITER = 5
_AUTO_LABEL_SHOW_MAX = 100


class AiLabeler:
    def __init__(self, model_path=_MODEL_PATH):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def detect_object(self, image_path, object_prompt="objects in the image", max_new_tokens=1024):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": f"""
                        Please describe this image. Extract the {object_prompt} and generate bounding box x_min, y_min, x_max, y_max in the format with as follows:
                        {{
                            "description": "",
                            "objects": 
                            [
                                {{
                                    "label": "",
                                    "bbox_2d": [x_min, y_min, x_max, y_max]
                                }},
                                {{
                                    "label": "",
                                    "bbox_2d": [x_min, y_min, x_max, y_max]
                                }},
                                ...
                            ]
                        }}
                        """
                    },
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # The generated_ids includes the inputs.input_ids prompt. Need to remove the prompt to get output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text
    
    def classify_image(self, image_path, prompt = "Please describe this image", max_new_tokens=1024):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": f"{prompt}"
                    },
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # The generated_ids includes the inputs.input_ids prompt. Need to remove the prompt to get output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text
    
    def extract_bbox(self, label_results):
        # Extract JSON content from the output text
        output_json = json.loads(label_results[0])
        objects = output_json['objects']
        logging.info(f"Detected {len(objects)} objects.")

        bboxs = []
        labels = []

        for object_item in objects:
            label = object_item['label']
            bbox = object_item['bbox_2d']

            labels.append(label)
            bboxs.append(bbox)
        
        return bboxs, labels
    
    def extract_classification(self, classification_result):
        output_json = json.loads(classification_result[0])
        plugin = output_json["Is plugged in"]
        connected_port = output_json["inserted to port (int)"]

        return plugin, connected_port

class CreateYoloDataset:
    def __init__(self, label_dict):
        self.label_dict = label_dict

    def create_yolo_label(self, bboxs, labels, image_size, file_name):
        string_to_write = ""
        for i, bbox in enumerate(bboxs):
            xmin, ymin, xmax, ymax = bbox
            image_height, image_width = image_size
            x_center_norm = (xmin + xmax) / 2.0 / image_width
            y_center_norm = (ymin + ymax) / 2.0 / image_height
            width_norm = (xmax - xmin) / image_width
            height_norm = (ymax - ymin) / image_height
            label_idx = self.label_dict.get(labels[i])
            string_to_write = string_to_write + f"{label_idx} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
            try:
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(string_to_write)
            except IOError as e:
                print(f"Error writing to file '{file_name}': {e}")

class CreateClassDataset:
    def __init__(self):
        pass

    def create_classification_label(self, plugin, classification_result, file_name):
        if plugin == False:
            label_idx = '0'
        else:
            label_idx = f"{classification_result}"
        try:
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(label_idx)
        except IOError as e:
            print(f"Error writing to file '{file_name}': {e}")

class PassFailDataset:
    def __init__(
        self,
        success_path_left,
        success_path_right,
        fail_path_left,
        fail_path_right,
        fail_fps,
        fail_duration,
        success_fps,
        left_mask_bbox,
        right_mask_bbox,
        val_ratio,
        output_path,
    ):
        self.success_path_left = success_path_left
        self.success_path_right = success_path_right
        self.fail_path_left = fail_path_left
        self.fail_path_right = fail_path_right
        self.fail_fps = fail_fps
        self.fail_duration = fail_duration
        self.success_fps = success_fps
        self.left_mask_bbox = left_mask_bbox
        self.right_mask_bbox = right_mask_bbox
        self.val_ratio = val_ratio
        self.output_path = output_path

        self.train_success_files_left = []
        self.train_success_files_right = []
        self.train_fail_files_left = []
        self.train_fail_files_right = []
        self.val_success_files_left = []
        self.val_success_files_right = []
        self.val_fail_files_left = []
        self.val_fail_files_right = []


    def _pair_video_files(self, left_dir, right_dir):
        def rename_files(file_dir):
            for file_name in os.listdir(file_dir):
                if file_name.endswith(".mp4"):
                    old_path = os.path.join(file_dir, file_name)
                    new_file_name = file_name.replace("_left", "").replace("_right", "")
                    new_path = os.path.join(file_dir, new_file_name)
                    if old_path != new_path:
                        os.rename(old_path, new_path)

        rename_files(left_dir)
        rename_files(right_dir)

        left_files = {
            file_name: os.path.join(left_dir, file_name)
            for file_name in sorted(os.listdir(left_dir))
            if file_name.endswith(".mp4")
        }
        right_files = {
            file_name: os.path.join(right_dir, file_name)
            for file_name in sorted(os.listdir(right_dir))
            if file_name.endswith(".mp4")
        }

        common_files = sorted(set(left_files) & set(right_files))
        if not common_files:
            raise ValueError(f"No paired mp4 files found in {left_dir} and {right_dir}")

        missing_left = sorted(set(right_files) - set(left_files))
        missing_right = sorted(set(left_files) - set(right_files))
        if missing_left:
            logging.warning("Missing left videos for: %s", missing_left)
        if missing_right:
            logging.warning("Missing right videos for: %s", missing_right)

        left_paths = [left_files[file_name] for file_name in common_files]
        right_paths = [right_files[file_name] for file_name in common_files]
        return left_paths, right_paths

    def _split_train_val(self, left_files, right_files):
        if len(left_files) != len(right_files):
            raise ValueError("left_files and right_files must have the same length")

        val_count = int(len(left_files) * self.val_ratio)
        if self.val_ratio > 0 and val_count == 0 and left_files:
            val_count = 1
    
        if val_count == 0:
            return left_files, right_files, [], []

        val_indices = set(random.sample(range(len(left_files)), val_count))
        train_left_files = []
        train_right_files = []
        val_left_files = []
        val_right_files = []

        for idx, (left_file, right_file) in enumerate(zip(left_files, right_files)):
            if idx in val_indices:
                val_left_files.append(left_file)
                val_right_files.append(right_file)
            else:
                train_left_files.append(left_file)
                train_right_files.append(right_file)

        return train_left_files, train_right_files, val_left_files, val_right_files

    def create_train_val_split(self):
        success_files_left, success_files_right = self._pair_video_files(
            self.success_path_left,
            self.success_path_right,
        )
        fail_files_left, fail_files_right = self._pair_video_files(
            self.fail_path_left,
            self.fail_path_right,
        )

        (
            self.train_success_files_left,
            self.train_success_files_right,
            self.val_success_files_left,
            self.val_success_files_right,
        ) = self._split_train_val(success_files_left, success_files_right)
        (
            self.train_fail_files_left,
            self.train_fail_files_right,
            self.val_fail_files_left,
            self.val_fail_files_right,
        ) = self._split_train_val(fail_files_left, fail_files_right)

        return (
            self.train_success_files_left,
            self.train_success_files_right,
            self.train_fail_files_left,
            self.train_fail_files_right,
            self.val_success_files_left,
            self.val_success_files_right,
            self.val_fail_files_left,
            self.val_fail_files_right,
        )

    def _get_video_duration(self, video_path):
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            raise IOError(f"ffprobe is not installed, cannot determine duration: {video_path}")

        command = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise IOError(f"Cannot determine video duration: {video_path}") from exc

        duration_text = result.stdout.strip()
        if not duration_text:
            raise ValueError(f"Empty duration returned for video: {video_path}")

        return float(duration_text)

    def get_success_duration_fail_episode_ratio(self):
        if not self.train_success_files_left and not self.train_fail_files_left:
            self.create_train_val_split()

        fail_episode_count = len(self.train_fail_files_left)
        if fail_episode_count == 0:
            raise ValueError("train_fail_files_left is empty, cannot compute ratio")

        success_duration = 0.0
        for left_video_path in self.train_success_files_left:
            success_duration += self._get_video_duration(left_video_path)

        return success_duration / fail_episode_count

    def _prepare_split_directories(self, split_name):
        images_dir = os.path.join(self.output_path, "images", split_name)
        labels_dir = os.path.join(self.output_path, "labels", split_name)

        for directory in (images_dir, labels_dir):
            if os.path.isdir(directory) and os.listdir(directory):
                for entry_name in os.listdir(directory):
                    entry_path = os.path.join(directory, entry_name)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
            os.makedirs(directory, exist_ok=True)

        return images_dir, labels_dir

    def _export_video_pairs(self, left_video_files, right_video_files, fps, label, image_idx, duration, split_name):
        images_dir = os.path.join(self.output_path, "images", split_name)
        labels_dir = os.path.join(self.output_path, "labels", split_name)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for left_video_path, right_video_path in zip(left_video_files, right_video_files):
            left_images = utils.convert_video_to_images(
                left_video_path,
                out_fps=fps,
                duration=duration,
            )
            right_images = utils.convert_video_to_images(
                right_video_path,
                out_fps=fps,
                duration=duration,
            )

            paired_count = min(len(left_images), len(right_images))
            if paired_count == 0:
                logging.warning(
                    "No paired images extracted from %s and %s",
                    left_video_path,
                    right_video_path,
                )
                continue

            if len(left_images) != len(right_images):
                logging.warning(
                    "Frame count mismatch for %s and %s; using %s paired frames",
                    left_video_path,
                    right_video_path,
                    paired_count,
                )

            for frame_idx in range(paired_count):
                file_stem = f"{image_idx:09d}"
                image_path = os.path.join(images_dir, f"{file_stem}.jpeg")
                label_path = os.path.join(labels_dir, f"{file_stem}.txt")
                left_image = utils.pad_vit_input(
                    left_images[frame_idx],
                    bbox=self.left_mask_bbox,
                )
                right_image = utils.pad_vit_input(
                    right_images[frame_idx],
                    bbox=self.right_mask_bbox,
                )

                utils.concat_images(
                    left_image,
                    right_image,
                    save_path=image_path,
                )
                with open(label_path, "w", encoding="utf-8") as file:
                    file.write(str(label))

                image_idx += 1

        return image_idx

    def create_train_classification_dataset(self):
        if not self.train_success_files_left and not self.train_fail_files_left:
            self.create_train_val_split()

        self._prepare_split_directories("train")
        image_idx = 1
        image_idx = self._export_video_pairs(
            self.train_success_files_left,
            self.train_success_files_right,
            fps=self.success_fps,
            label=0,
            image_idx=image_idx,
            duration=-1,
            split_name="train",
        )
        image_idx = self._export_video_pairs(
            self.train_fail_files_left,
            self.train_fail_files_right,
            fps=self.fail_fps,
            label=1,
            image_idx=image_idx,
            duration=self.fail_duration,
            split_name="train",
        )

        return image_idx - 1

    def create_val_classification_dataset(self):
        if not self.val_success_files_left and not self.val_fail_files_left:
            self.create_train_val_split()

        self._prepare_split_directories("val")
        image_idx = 1
        image_idx = self._export_video_pairs(
            self.val_success_files_left,
            self.val_success_files_right,
            fps=self.success_fps,
            label=0,
            image_idx=image_idx,
            duration=-1,
            split_name="val",
        )
        image_idx = self._export_video_pairs(
            self.val_fail_files_left,
            self.val_fail_files_right,
            fps=self.fail_fps,
            label=1,
            image_idx=image_idx,
            duration=self.fail_duration,
            split_name="val",
        )

        return image_idx - 1

@retry(stop_max_attempt_number=5, wait_fixed=100)
def yolo_autolabel(Labeler, Yolodataset, path, file_name, prompt, max_new_tokens=2048):
    result = Labeler.detect_object(path, prompt, max_new_tokens=max_new_tokens)
    try:
        bboxs, labels = Labeler.extract_bbox(result)
        Yolodataset.create_yolo_label(
            bboxs, 
            labels, 
            (1000, 1000),  # (height, width)
            file_name
        )
    except:
        raise Exception("Inference error")
    
    return result, bboxs, labels

@retry(stop_max_attempt_number=5, wait_fixed=100)
def classification_autolabel(Labeler, Classdataset, path, prompt, file_name, max_new_tokens=2048):
    result = Labeler.classify_image(path, prompt, max_new_tokens=max_new_tokens)
    try:
        plugin, connected_port = Labeler.extract_classification(result)
        if file_name is not None:
            Classdataset.create_classification_label(
                plugin, 
                connected_port,
                file_name
            )
    except:
        raise Exception("Inference error")
    
    return result, plugin, connected_port

def create_label(time_labels, timestamp, file_name):
    for key in time_labels:
        if timestamp >= key:
            port_label = time_labels[key]
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(f"{port_label}")
    except IOError as e:
        logging.info(f"Error writing to file '{file_name}': {e}")

def create_video(root_dir, trgt_video_file, fps=30):
    path_list = utils.create_file_list(root_dir)
    path_list = sorted(path_list)
    frame = Image.open(path_list[0]).convert("RGB")
    width, height = frame.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(trgt_video_file, fourcc, fps, (width, height))

    for path in path_list:
        timestamp = path.split("/")[-1].replace(".jpg", "")
        frame = Image.open(path).convert("RGB")
        image_edit = utils.add_text_2_img(frame, timestamp, font_size=30, xy=(20, 20), color=(0, 0, 255))
        image_buffer = io.BytesIO(image_edit)
        frame_edit = Image.open(image_buffer)
        # Convert PIL image (RGB) to NumPy array and then to OpenCV format (BGR)
        numpy_img = np.array(frame_edit)
        opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        video.write(opencv_img)

    video.release()
    
def batch_label(root_dir, root_label, trgt_dir_img, trgt_dir_label, step, init_idx=0):
    path_list = utils.create_file_list(root_dir)
    path_list = sorted(path_list)
    time_labels = {}
    logging.info(root_dir)
    with open(root_label, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        timestamp, port = str(line).strip("\n").split("\t")
        time_labels[timestamp] = int(port)

    for path in path_list[init_idx::step]:
        img_file_name = path.replace(root_dir, trgt_dir_img)
        label_file_name = path.replace(root_dir, trgt_dir_label).replace(".jpg", ".txt")
        timestamp = path.split("/")[-1].replace(".jpg", "")
        shutil.copyfile(path, img_file_name)
        create_label(time_labels, timestamp, label_file_name)

class LabelFSM:
    def __init__(self, filter_frame_up = 1, filter_frame_down = 2):
        self.state = 0  # Initial state
        self.state_lst = 0 # Initial state memory
        self.filter_frame_up = filter_frame_up
        self.filter_frame_down = filter_frame_down
        self.prediction_history_up = [0] * self.filter_frame_up # To store recent predictions for filtering
        self.prediction_history_down = [0] * self.filter_frame_down # To store recent predictions for filtering
        self.timer = 0
        
    def get_state_info(self):
        return self.state, self.state_lst, self.prediction_history_up, self.prediction_history_down
    
    def get_state_timer(self):
        return self.timer
    
    def transition(self, prediction, dt=None):
        self.state_lst = self.state
        # Define your state transition logic here
        if self.state == 0:
            if all(predict == 1 for predict in self.prediction_history_up) and prediction == 1:
                self.state = 1
            elif all(predict == 2 for predict in self.prediction_history_up) and prediction == 2:
                self.state = 2
            elif all(predict == 3 for predict in self.prediction_history_up) and prediction == 3:
                self.state = 3
            elif all(predict == 4 for predict in self.prediction_history_up) and prediction == 4:
                self.state = 4
            elif all(predict == 5 for predict in self.prediction_history_up) and prediction == 5:
                self.state = 5
            else:
                self.state = 0
        elif self.state == 1:
            if all(predict != 1 for predict in self.prediction_history_down) and prediction != 1:
                self.state = 0
            else:
                self.state = 1
        elif self.state == 2:
            if all(predict != 2 for predict in self.prediction_history_down) and prediction != 2:
                self.state = 0
            else:
                self.state = 2
        elif self.state == 3:
            if all(predict != 3 for predict in self.prediction_history_down) and prediction != 3:
                self.state = 0
            else:
                self.state = 3
        elif self.state == 4:
            if all(predict != 4 for predict in self.prediction_history_down) and prediction != 4:
                self.state = 0
            else:
                self.state = 4
        elif self.state == 5: 
            if all(predict != 5 for predict in self.prediction_history_down) and prediction != 5:
                self.state = 0
            else:
                self.state = 5

        self.prediction_history_up.pop(0)
        self.prediction_history_up.append(prediction)
        self.prediction_history_down.pop(0)
        self.prediction_history_down.append(prediction)

        return self.state

def read_image_rgb(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def cal_image_diff(image_path_1, image_path_2):
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM between the two images
    # score is the SSIM metric, diff is the difference image
    (score, _) = ssim(gray1, gray2, full=True)

    return score
    
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
        result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, label_prompt, file_name, max_new_tokens=max_new_tokens)
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
        result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, label_prompt, file_name, max_new_tokens=max_new_tokens)
        if i % _AUTO_LABEL_SHOW_ITER == 0 and i < _AUTO_LABEL_SHOW_MAX:
            logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")
        logging.info(f"Saved label file as {file_name}.")

def classifier_autolabel_complex(train_image_dir, train_label_dir, val_image_dir, val_label_dir, label_prompt, step = 5, max_new_tokens=1024):
    db_result_train = pd.DataFrame(columns=["image_path", "description", "is_plugged_in", "connected_port", "delta_rgb"])
    db_result_val = pd.DataFrame(columns=["image_path", "description", "is_plugged_in", "connected_port", "delta_rgb"])
    classifier = AiLabeler(_MODEL_PATH)
    ClassLabel = CreateClassDataset()

    # create train data set
    root_dir = train_image_dir
    path_list = create_file_list(root_dir)
    path_list = sorted(path_list)

    for i, path in enumerate(path_list):
        if i % step == 0:
            logging.info(f"Processing image: {path}")
            result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, label_prompt, file_name=None, max_new_tokens=max_new_tokens)
            diff_rgb = cal_image_diff(path, path_list[i-1]) if i > 0 else 0
            db_result_train = pd.concat([db_result_train, pd.DataFrame({"image_path": [path], "description": [result], "is_plugged_in": [plugin], "connected_port": [connected_port], "delta_rgb": [diff_rgb]})], ignore_index=True)
            if i % _AUTO_LABEL_SHOW_ITER == 0 and i < _AUTO_LABEL_SHOW_MAX:
                logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")

    # create val data set
    root_dir = val_image_dir
    path_list = create_file_list(root_dir)
    path_list = sorted(path_list)
    for i, path in enumerate(path_list):
        if i % step == 0:
            logging.info(f"Processing image: {path}")
            result, plugin, connected_port = classification_autolabel(classifier, ClassLabel, path, label_prompt, file_name=None, max_new_tokens=max_new_tokens)
            diff_rgb = cal_image_diff(path, path_list[i-1]) if i > 0 else 0
            db_result_val = pd.concat([db_result_val, pd.DataFrame({"image_path": [path], "description": [result], "is_plugged_in": [plugin], "connected_port": [connected_port], "delta_rgb": [diff_rgb]})], ignore_index=True)
            if i % _AUTO_LABEL_SHOW_ITER == 0 and i < _AUTO_LABEL_SHOW_MAX:
                logging.info(f"Description: {result}, Is Plugged In: {plugin}, Connected to Port: {connected_port}")

    # Process raw result
    db_result_train['label'] = db_result_train.apply(lambda row: 0 if row['is_plugged_in'] == False or row['delta_rgb'] < 0.94 else row['connected_port'], axis=1)
    port_fsm = LabelFSM()
    for i in range(len(db_result_train)):
        current_prediction = db_result_train.loc[i, 'label']
        filtered_state = port_fsm.transition(current_prediction)
        db_result_train.loc[i, 'label_filtered'] = filtered_state

    db_result_val['label'] = db_result_val.apply(lambda row: 0 if row['is_plugged_in'] == False or row['delta_rgb'] < 0.94 else row['connected_port'], axis=1)
    port_fsm = LabelFSM()
    for i in range(len(db_result_val)):
        current_prediction = db_result_val.loc[i, 'label']
        filtered_state = port_fsm.transition(current_prediction)
        db_result_val.loc[i, 'label_filtered'] = filtered_state
            
    with open(train_label_dir, 'wb') as file:
        for i in range(len(db_result_train)):
            if i == 0 or db_result_train.loc[i, 'label_filtered'] != db_result_train.loc[i-1, 'label_filtered']:
                time_str = db_result_train.loc[i, 'image_path'].replace(".jpg", "")
                file.write(f"{time_str.split('/')[-1]}\t{db_result_train.loc[i, 'label_filtered']}\n".encode())
    with open(val_label_dir, 'wb') as file:
        for i in range(len(db_result_val)):
            if i == 0 or db_result_val.loc[i, 'label_filtered'] != db_result_val.loc[i-1, 'label_filtered']:
                time_str = db_result_val.loc[i, 'image_path'].replace(".jpg", "")
                file.write(f"{time_str.split('/')[-1]}\t{db_result_val.loc[i, 'label_filtered']}\n".encode())

if __name__ == "__main__":
    ObjDetectLabeler = AiLabeler(_MODEL_PATH)
    CreateYoloLabel = CreateYoloDataset({"circular port": 0, "rectangular port": 1})

    # create train data set
    root_dir = "/home/yang/MyRepos/tensorRT/datasets/port0/images/train"
    trgt_dir = "/home/yang/MyRepos/tensorRT/datasets/port0/labels/train"
    path_list = utils.create_file_list(root_dir)

    for path in path_list:
        file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
        logging.info(f"Processing image: {path}")
        result, bboxs, labels = yolo_autolabel(ObjDetectLabeler, CreateYoloLabel, path, file_name, "circular ports on the white board", max_new_tokens=2048)
        utils.draw_bbox(
            path,
            bboxs[:],
            labels[:],
            new_size = (1000, 1000)
        )
        logging.info(f"Saved label file as {file_name}.")

    # create val data set
    root_dir = "/home/yang/MyRepos/tensorRT/datasets/port0/images/val"
    trgt_dir = "/home/yang/MyRepos/tensorRT/datasets/port0/labels/val"
    path_list = utils.create_file_list(root_dir)

    for path in path_list:
        file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
        logging.info(f"Processing image: {path}")
        result, bboxs, labels = yolo_autolabel(ObjDetectLabeler, CreateYoloLabel, path, file_name, "circular ports on the white board", max_new_tokens=2048)
        utils.draw_bbox(
            path,
            bboxs[:],
            labels[:],
            new_size = (1000, 1000)
        )
        logging.info(f"Saved label file as {file_name}.")
    plt.show()