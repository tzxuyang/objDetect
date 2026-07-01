# Simulate robot monitor model
* update monitor config under `monitor_config_pnp.json`, `monitor_config_port.json`, or `monitor_config_ioboard.json`

* run monitor node
  * IO board / P548
  ```
  uv run monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_ioboard.json --print_logs
  uv run monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_p548.json --print_logs
  ```

* run camera simulation node, which replays recorded left/right videos
  ```
  * IO board without saving images
  ```
  uv run monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_ioboard.json
  uv run monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_p548.json
  ```
  * IO board with image saving enabled
  ```
  uv run monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_ioboard.json --save_image
  uv run monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_p548.json --save_image
  ```

* batch test multiple videos
  ```
  uv run monitor_app/src/test_monitor.py
  ```
