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
python yolo_detect.py --mode autolabel --train_image /home/yang/MyRepos/tensorRT/datasets/port0/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port0/labels/train \
    --val_image /home/yang/MyRepos/tensorRT/datasets/port0/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port0/labels/val
```

* Step 2: train
```
python yolo_detect.py --mode train --dataset ./data_configs/port0.yaml
```

* Step 3: predict
```
python yolo_detect.py --mode predict --image ./images/circular_port_22.jpg --checkpoint ./runs/detect/train/weights/best.pt
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
python src/convert_img.py --root_dir /home/yang/datasets/white_board_image2/
```

* Step 1: autolabel
```
python status_classifier.py --mode autolabel --train_image /home/yang/MyRepos/tensorRT/datasets/port_cls/images/train --train_label /home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train --val_image /home/yang/MyRepos/tensorRT/datasets/port_cls/images/val --val_label /home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val
```

* Step 2: train
Change the training config under /data_configs/train_config.json and run CLI:
```
python status_classifier.py --mode train --project_name classifier_dinov3_small_no_augment
```
The trained checkpoint will be saved in /runs/

* Step 3: predict
```
python status_classifier.py --mode predict --checkpoint ./checkpoints/dino_classifier.pth --image ./images/port_2.jpg
```