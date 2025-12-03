import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
import re
import torch
import json
import logging
from retrying import retry
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import utils

logging.basicConfig(level=logging.INFO)

_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

class ObjectDetector:
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

@retry(stop_max_attempt_number=5, wait_fixed=100)
def autolabel(path, prompt, max_new_tokens=2048):
    result = ObjDetectLabeler.detect_object(path, prompt, max_new_tokens=max_new_tokens)
    try:
        bboxs, labels = ObjDetectLabeler.extract_bbox(result)
        CreateYoloLabel.create_yolo_label(
            bboxs, 
            labels, 
            (1000, 1000),  # (height, width)
            file_name
        )
    except:
        raise Exception("Inference error")
    
    return result, bboxs, labels

if __name__ == "__main__":
    ObjDetectLabeler = ObjectDetector(_MODEL_PATH)
    CreateYoloLabel = CreateYoloDataset({"circular port": 0, "rectangular port": 1})

    # create train data set
    root_dir = "/home/yang/MyRepos/tensorRT/datasets/port0/images/train"
    trgt_dir = "/home/yang/MyRepos/tensorRT/datasets/port0/labels/train"
    path_list = utils.create_file_list(root_dir)

    for path in path_list:
        file_name = path.replace(root_dir, trgt_dir).replace(".jpg", ".txt")
        logging.info(f"Processing image: {path}")
        result, bboxs, labels = autolabel(path, "circular ports on the white board", max_new_tokens=2048)
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
        result, bboxs, labels = autolabel(path, "circular ports on the white board", max_new_tokens=2048)
        utils.draw_bbox(
            path,
            bboxs[:],
            labels[:],
            new_size = (1000, 1000)
        )
        logging.info(f"Saved label file as {file_name}.")
    plt.show()

    # image_path = "/home/yang/MyRepos/object_detection/images/dog4.jpg"
    # image_path = "/home/yang/datasets/visual_image/edited_image_5.jpg"
    # # result = ObjDetectLabeler.detect_object(image_path, "objects in the image")
    # result = ObjDetectLabeler.detect_object(image_path, "circular ports on the white board")

    # bboxs, labels = ObjDetectLabeler.extract_bbox(result)

    # a, b, c, d = [0, 1, 2, 3]  # Dummy to avoid syntax error
    # print(a)
    # print("Bounding Boxes:", bboxs)
    # print("Labels:", labels)

    # utils.draw_bbox(
    #     image_path,
    #     bboxs[:],
    #     labels[:],
    #     new_size = (1000, 1000)
    # )