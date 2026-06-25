import json
from re import sub
import subprocess
import time


success_video = ["episode_000001",
                 "episode_000003",
                 "episode_000014",
                 "episode_000019",
                 "episode_000020",
                 "episode_000024",
                 "episode_000025",
                 "episode_000061",
                 "episode_000073"]
fail_video = ["episode_000004",
              "episode_000006",
              "episode_000011",
              "episode_000015",
              "episode_000017",
              "episode_000018",
              "episode_000059"]

_ROOT_DIR = "/home/yang/MyRepos/object_detection/videos"

def update_json(json_file, key, value):
    with open(json_file, 'r') as f:
        data = json.load(f)
    data[key] = value
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def run_monitor_camera_nodes(file_name=None):
    if file_name is not None:
        update_json("data_configs/monitor_config_ioboard.json", "video_path_left", f'{_ROOT_DIR}/{file_name}_left.mp4')
        update_json("data_configs/monitor_config_ioboard.json", "video_path_right", f'{_ROOT_DIR}/{file_name}_right.mp4')
    term1 = subprocess.Popen(["uv", "run", "python", "monitor_app/src/monitor_node.py", "--config_path", "data_configs/monitor_config_ioboard.json"], stdout=subprocess.PIPE)
    time.sleep(1)
    term2 = subprocess.Popen(["uv", "run", "python", "monitor_app/src/camera_sim_node.py", "--config_path", "data_configs/monitor_config_ioboard.json"])
    time.sleep(30)
    for child in term2.children(recursive=True):
        child.terminate()
    for child in term1.children(recursive=True):
        child.kill()
    term2.kill()
    time.sleep(0.01)
    term1.kill()
    print("----------------------------------------------------------------------------------------------")
    for line in term1.stdout:
        if 'Published monitor warning:' in line.decode('utf-8'):
            print(f"Captured: {line.strip().decode('utf-8'  )}")

if __name__ == "__main__":
    run_monitor_camera_nodes('episode_000001')