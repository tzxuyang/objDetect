# Simulate robot monitor model
* update monitor config under `monitor_config_pnp.json`, `monitor_config_port.json`, or `monitor_config_ioboard.json`

* run monitor node
  * IO board
  ```
  uv run python monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_ioboard.json
  ```

* run camera simulation node, which replays recorded left/right videos
  ```
  * IO board without saving images
  ```
  uv run python monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_ioboard.json
  ```
  * IO board with image saving enabled
  ```
  uv run python monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_ioboard.json --save_image
  ```