# Simulate robot monitor model
* update monitor config under monitor_config_pnp.json or monitor_config_port.json

* run monitor node
  * Pick and place
  ```
  uv run python monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_pnp.json
  ```
  * Port insert
  ```
  uv run python monitor_app/src/monitor_node.py --config_path data_configs/monitor_config_port.json
  ```

* run camera simulation node, which replay a recorded video
  * Pick and place
  ```
  uv run python monitor_app/src/camera_sim_node.py --config-path data_configs/monitor_config_pnp.json
  ```
  * Port insert
  ```
  uv run python monitor_app/src/camera_sim_node.py --config_path data_configs/monitor_config_port.json
  ```