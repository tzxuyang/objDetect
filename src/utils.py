import os
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from bs4 import BeautifulSoup, Tag
from pathlib import Path
import re
import transformers
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_file_list(root_dir):
    dir_contents = os.listdir(root_dir)
    path_list = [os.path.join(root_dir, file_path) for file_path in dir_contents]

    return path_list

def read_image(path):
    img = Image.open(path)

    img.show()

    print(f"Image size: {img.size}")

def process_image(path):
    img = Image.open(path)

    width, height = img.size
    if width < height:
        img = img.rotate(angle=90, resample=Image.BICUBIC, expand=1)
    if width > 2000:
        img = img.resize((1440, 1080))
    img.show()

    print(f"Image size: {img.size}")

    return img

# register_heif_opener()
def convert_heic_to_jpeg(heic_path, jpeg_path):
    try:
        # Open the HEIC image
        img = Image.open(heic_path)

        # Convert to RGB mode if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        img = img.resize((640, 480))
        
        # Save the image in JPEG format
        img.save(jpeg_path, "JPEG", quality=95)

        print(f"Successfully converted {heic_path} to {jpeg_path}")

    except Exception as e:
        print(f"Error converting {heic_path}: {e}")

def draw_bbox(image_path, bboxs, labels, new_size = (1000, 600)):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Resize the image
    image = image.resize(new_size)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label in zip(bboxs, labels):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min),
            width,
            height,
            linewidth = 2,
            edgecolor = 'r',
            facecolor = 'none'
        )
        ax.add_patch(rect)
        plt.text(
            x_min, 
            y_min, 
            f"{label}", 
            color='white', 
            fontsize=12,
            bbox = dict(facecolor='red', alpha=0.5)
        )

    plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    root_dir = "/home/yang/datasets/visual_image"
    path_list = create_file_list(root_dir)

    for path in path_list:
        img = process_image(path)
        img.save("edited_" + os.path.basename(path))