from utils import convert_heic_to_jpeg, create_file_list
from pillow_heif import register_heif_opener
import logging
import tyro
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

register_heif_opener()

_OFFSET = 193

@dataclass
class Config:
    root_dir: str = "/home/yang/datasets/white_board_image3/"  # Directory containing HEIC images

if __name__ == "__main__":
    config = tyro.cli(Config)
    root_directory = config.root_dir
    file_list = create_file_list(root_directory)
    file_list_new = []
    for file in file_list:
        file_short = file.split("/")[-1]
        if not file_short.startswith("._"):
            file_list_new.append(file)

    logging.info(len(file_list_new))
    logging.info(file_list_new)

    for i, file in enumerate(file_list_new):
        num = i + _OFFSET
        if num < 10:
            jpeg_image_path = root_directory + f"00000000000{num}.jpg"
        elif num < 100:
            jpeg_image_path = root_directory + f"0000000000{num}.jpg"
        elif num < 1000:
            jpeg_image_path = root_directory + f"000000000{num}.jpg"
        else:
            jpeg_image_path = root_directory + f"00000000{num}.jpg"
        convert_heic_to_jpeg(file, jpeg_image_path, img_size = (848, 480))


