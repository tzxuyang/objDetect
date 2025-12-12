from src.utils import convert_heic_to_jpeg, create_file_list
from pillow_heif import register_heif_opener

register_heif_opener()

if __name__ == "__main__":
    root_directory = "/home/yang/datasets/white_board_image/"
    file_list = create_file_list(root_directory)
    file_list_new = []
    for file in file_list:
        file_short = file.split("/")[-1]
        if not file_short.startswith("._"):
            file_list_new.append(file)

    print(len(file_list_new))
    print(file_list_new)

    for i, file in enumerate(file_list_new):
        if i < 10:
            jpeg_image_path = root_directory + f"00000000000{i}.jpg"
        else:
            jpeg_image_path = root_directory + f"0000000000{i}.jpg"
        convert_heic_to_jpeg(file, jpeg_image_path)


