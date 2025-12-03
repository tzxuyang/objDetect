import os
from ultralytics import YOLO
from utils import draw_bbox
import matplotlib.pyplot as plt

_IMG_PATH = "./images/circular_port_22.jpg"

class YOLOCustom:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def load_model(self, weights_path: str):
        self.model = YOLO(weights_path)
        return self.model

    def train(self, data: str, epochs: int, imgsz: int):
        results = self.model.train(data=data, epochs=epochs, imgsz=imgsz)
        return results
    
    def predict(self, img_path: str, conf: float = 0.25):
        results = self.model(source=img_path, conf=conf)
        return results
    
if __name__ == "__main__":
    # # model("port0/weights/best.pt").export(format="onnx")  # export the best model to ONNX format
    folders = os.listdir("./runs/detect")
    if len(folders) > 1:
        folder_cnt = int(max([(f.replace("train", "")) for f in folders if "train" in f]))
        next_folder = f"train{folder_cnt + 1}"
    elif len(folders) == 0:
        next_folder = "train"
    else:
        next_folder = "train2"

    # train custom model using YOLOCustom class
    yolo_port = YOLOCustom("./checkpoints/yolov8n.pt")
    yolo_port.train(data="./data_configs/port0.yaml", epochs=100, imgsz=640)

    # predict using the trained model
    yolo_port.load_model(f"./runs/detect/{next_folder}/weights/best.pt")
    results = yolo_port.predict(img_path=_IMG_PATH, conf=0.25)

    height, width = results[0].orig_shape
    label_dict = results[0].names
    labels = [label_dict[cls] for cls in results[0].boxes.cls.tolist()]
    bboxs = results[0].boxes.xyxy.tolist()
    draw_bbox(
        _IMG_PATH,
        bboxs[:],
        labels[:],
        new_size = (width, height)
    )
    plt.show()