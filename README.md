# YOLO object detection tool
---
This yolo object detection tool leverages a yolo11 backbone. It is used to do customized object detection, while leveraging Qwen3-4B/Qwen3-8B pretrained model capability

## Tutorials
Under examples folder, there are tutorials to use qwen and a pretrained yolo model to detect objects in an image, generating class and bounding box
* object_detect_yolo.ipynb: use a pretrained yolo model ("hustvl/yolos-tiny") to detect objects
* object_detection_qwen.ipynb: use Qwen VLM model to detect objects in an image

## Customized object detection model
The customed object detection model create a pipeline, where a Qwen pretrained model is used to auto-label images.
The autolabel results are stored in a specified dir path in a format used by YOLO lib
Then the yolo8n model architecture is selected and trained with auto-labeled dataset
Finally the tool can be used to predict (detect) the customized object class

* Step 1: autolabel
```
uv run python yolo_detect.py --mode autolabel --train_image /home/yang/MyRepos/tensorRT/datasets/port0/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port0/labels/train \
    --val_image /home/yang/MyRepos/tensorRT/datasets/port0/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port0/labels/val
```

* Step 2: train
```
uv run python yolo_detect.py --mode train --dataset ./data_configs/port0.yaml
```

* Step 3: predict
```
uv run python yolo_detect.py --mode predict --image ./images/circular_port_22.jpg --checkpoint ./runs/detect/train/weights/best.pt
```

# DinoV3 classification tool
---
This tool does customized classification task. It is built upon a dinov3 (20M parameter) ViT backbone. It also leverages Qwen3-4B/Qwen3-8B pretrained model to do auto-labeling and train the dinov3-classifier with the autobabeled image-label pairs.

## Customized image classification model
The customed object detection model create a pipeline, where a Qwen pretrained model is used to auto-label images.
The autolabel results are stored in a specified dir path in a format used by YOLO lib
The dinov3 backbone is selected and customized with MLP head. It is then trained with auto-labeled dataset
Finally the tool can be used to classify the image into predefined labels

* Step 0: image format conversion (optional, only for image token from iphone)
```
uv run python src/convert_img.py --root_dir /home/yang/datasets/white_board_image2/
```

* Step 1.1: autolabel
```
uv run status_classifier.py --mode autolabel --label_config data_configs/train_config_ioboard.json
```

* Step 2: train
Create a train config file such as /data_configs/train_config_port.json under /data/configs. Update the training config and run CLI:
```
# for ioboard classification 
uv run python status_classifier.py --mode train --train_config data_configs/train_config_ioboard.json --project_name classifier_ioboard_0622
```
The trained checkpoint will be saved in /runs/

* Step 3: predict
```
# for ioboard classification
uv run status_classifier.py --mode predict --train_config data_configs/train_config_ioboard.json --checkpoint ./checkpoints/dino_classifier.pth --image-left ./images/camera_left_000017.jpg --image-right ./images/camera_right_000017.jpg
```