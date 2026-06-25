import os

from src.utils import convert_video_to_images, concat_images, pad_vit_input

padding_bbox_right = [128, 240, 512, 480]
padding_bbox_left =  [-1, -1, -1, -1]

if __name__ == "__main__":
    video_path_left = "/home/yang/MyRepos/tensorRT/datasets/pick_n_place_fail/left/episode_000004_00.mp4"  # Replace with your video path
    video_path_right = "/home/yang/MyRepos/tensorRT/datasets/pick_n_place_fail/right/episode_000004_00.mp4"  # Replace with your video path
    output_path = "./dataset/images/train"  # Directory to save extracted frames
    out_fps = 30                      # Number of frames per second to extract

    try:
        images_left = convert_video_to_images(video_path_left, out_fps, duration=1)
        images_right = convert_video_to_images(video_path_right, out_fps, duration=1)
        print(f"Extracted {len(images_left)} images from the left video.")
        print(f"Extracted {len(images_right)} images from the right video.")
        images_left_padded = [pad_vit_input(img, bbox=padding_bbox_left) for img in images_left]
        images_right_padded = [pad_vit_input(img, bbox=padding_bbox_right) for img in images_right]
        os.makedirs(output_path, exist_ok=True)
        paired_count = min(len(images_left_padded), len(images_right_padded))
        for idx in range(paired_count):
            save_path = os.path.join(output_path, f"concat_{idx:06d}.jpg")
            concat_images(
                images_left_padded[idx],
                images_right_padded[idx],
                padding=0,
                save_path=save_path,
            )

    except Exception as e:
        print(f"An error occurred: {e}")