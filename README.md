# Object detection tool

# Tutorials
Under examples folder, there are tutorials to use qwen and a pretrained yolo model to detect objects in an image, generating class and bounding box
* object_detect_yolo.ipynb: use a pretrained yolo model ("hustvl/yolos-tiny") to detect objects
* object_detection_qwen.ipynb: use Qwen VLM model to detect objects in an image

# Customed object detection model
The customed object detection model create a pipeline, where a Qwen pretrained model is used to auto-label images.
The autolabel results are stored in a specified dir path in a format used by YOLO lib
Then the yolo8n model architecture is selected and trained with auto-labeled dataset
Finally the tool can be used to predict (detect) the customed object class

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
