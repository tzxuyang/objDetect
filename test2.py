import json

from src.auto_labeling import PassFailDataset

if __name__ == "__main__":
    with open("./data_configs/train_config_ioboard.json", "r", encoding="utf-8") as file:
        train_config = json.load(file)

    dataset = PassFailDataset(
        success_path_left="/home/yang/MyRepos/tensorRT/datasets/pick_n_place_success/left",
        success_path_right="/home/yang/MyRepos/tensorRT/datasets/pick_n_place_success/right",
        fail_path_left="/home/yang/MyRepos/tensorRT/datasets/pick_n_place_fail/left",
        fail_path_right="/home/yang/MyRepos/tensorRT/datasets/pick_n_place_fail/right",
        fail_fps=60,
        success_fps=1,
        left_mask_bbox=train_config["padding_bbox_left"],
        right_mask_bbox=train_config["padding_bbox_right"],
        val_ratio=0.2,
        fail_duration=0.5,
        output_path="./dataset"
    )
    dataset.create_train_val_split()

    ratio = dataset.get_success_duration_fail_episode_ratio()
    print(f"Success duration to fail episode ratio: {ratio:.2f}")

    print("Train sucess files:------------------------------------------------------")
    print(dataset.train_success_files_left)
    print("Train fail files:------------------------------------------------------")
    print(dataset.train_fail_files_left)
    print("Validation success files:------------------------------------------------------")
    print(dataset.val_success_files_left)
    print("Validation fail files:------------------------------------------------------")
    print(dataset.val_fail_files_left)

    dataset.create_train_classification_dataset()
    dataset.create_val_classification_dataset()