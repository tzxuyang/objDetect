import json
import subprocess
import sys
import time
import tyro
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rcl_interfaces.msg import Log

@dataclass
class Config:
    test_case: str

_ROOT_DIR = "/home/yang/MyRepos/object_detection/videos"
_WARNING_TOKEN = "Published monitor warning: True"
_RUN_TIMEOUT = 40.0  # max seconds to run each episode

# Expected monitor-warning outcome per category. A warning flags a detected
# problem, so failed episodes should warn and successful ones should stay clean.
# Flip these if your monitor polarity is the other way around.
_EXPECT_WARNING_SUCCESS = False
_EXPECT_WARNING_FAIL = True

def update_json(json_file, key, value):
    with open(json_file, 'r') as f:
        data = json.load(f)
    data[key] = value
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


class RosoutWatcher(Node):
    """Subscribes to /rosout and flags when a target string is logged.

    monitor_node logs via self.get_logger(), which rclpy publishes to the
    /rosout topic as rcl_interfaces/msg/Log. Reading the warning here means we
    never touch the child process's stdout/stderr at all.
    """

    def __init__(self, token):
        super().__init__('rosout_watcher')
        self._token = token
        self._found = False

        # Match the rosout publisher QoS (reliable, keep-last/1000). We create
        # this subscription once, before any node launches, so VOLATILE is
        # enough and it avoids historical re-delivery across episodes.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(Log, '/rosout', self._on_log, qos)

    def _on_log(self, msg: Log):
        # msg.name = logger/node name, msg.msg = the logged text, msg.level =
        # DEBUG(10)/INFO(20)/WARN(30)/ERROR(40)/FATAL(50).
        if self._token in msg.msg:
            self._found = True

    @property
    def found(self):
        return self._found

    def reset(self):
        self._found = False


def _drain(watcher, seconds):
    """Spin the watcher for a short while to flush pending /rosout callbacks."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(watcher, timeout_sec=0.05)


def run_monitor_camera_nodes(watcher, config_path, file_name=None):
    """Run the monitor + camera nodes for one episode.

    Returns True if monitor_node logged the warning token to /rosout.
    """
    if file_name is not None:
        update_json(config_path,
                    "video_path_left", f'{_ROOT_DIR}/{file_name}_left.mp4')
        update_json(config_path,
                    "video_path_right", f'{_ROOT_DIR}/{file_name}_right.mp4')

    watcher.reset()

    # We no longer pipe stdout; logs come through /rosout. Silence the consoles
    # to keep the batch output readable (drop these kwargs to see node logs).
    term1 = subprocess.Popen(
        ["uv", "run", "python", "monitor_app/src/monitor_node.py",
         "--config_path", config_path, "--print_logs"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    term2 = subprocess.Popen(
        ["uv", "run", "python", "monitor_app/src/camera_sim_node.py",
         "--config_path", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Spin the watcher while the nodes run; stop early once the warning shows up
    # or once the camera node finishes playing (play-once).
    deadline = time.monotonic() + _RUN_TIMEOUT
    while time.monotonic() < deadline and not watcher.found:
        rclpy.spin_once(watcher, timeout_sec=0.1)
        if term2.poll() is not None:
            break

    # --- Clean teardown: terminate first, fall back to kill if needed. ---
    for proc in (term2, term1):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    # Flush any warning that was logged right before shutdown.
    _drain(watcher, 0.5)
    return watcher.found


def _run_category(watcher, config_path, label, episodes, expect_warning, results):
    for name in episodes:
        print("=" * 80)
        print(f"[{label}] Running episode: {name}  (expect warning={expect_warning})")
        found = run_monitor_camera_nodes(watcher, config_path, name)
        passed = (found == expect_warning)
        results.append((label, name, found, expect_warning, passed))
        verdict = "PASS" if passed else "FAIL"
        print(f"  -> warning published: {found}  [{verdict}]")

def main(cfg: Config)-> None:
    test_case = cfg.test_case

    if test_case == "ioboard":
        _WARNING_FILTER_DURATION_SUCCESS = 0.2  # seconds to filter out repeated warnings 0.2
        _WARNING_FILTER_DURATION_FAIL = 0.05  # seconds to filter out repeated warnings 0.05
        config_path = "data_configs/monitor_config_ioboard.json"
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
    elif test_case == "p548":
        _WARNING_FILTER_DURATION_SUCCESS = 2.0  # seconds to filter out repeated warnings 2.0
        _WARNING_FILTER_DURATION_FAIL = 2.0  # seconds to filter out repeated warnings 2.0
        config_path = "data_configs/monitor_config_p548.json"
        success_video = ["000050_1",
                         "000051_1",
                         "000100_1",
                         "000101_1",
                         "000200_1",
                         "000201_1",
                         "000300_1",
                         "000311_1",]
        fail_video = ["000350_0",
                      "000400_0",
                      "000403_0",
                      "000449_0",
                      "000450_0"]
    else:
        raise ValueError(f"Unknown test_case: {test_case}")

    # Run me with `uv run python ...` so rclpy is on the path and the
    # ROS_DOMAIN_ID matches the child node processes (they inherit this env).
    rclpy.init()
    watcher = RosoutWatcher(_WARNING_TOKEN)
    # results: list of (label, name, found, expected, passed)
    results = []
    try:
        update_json(config_path, "warning_filter_duration", _WARNING_FILTER_DURATION_SUCCESS)
        _run_category(watcher, config_path, "success", success_video,
                      _EXPECT_WARNING_SUCCESS, results)
        update_json(config_path, "warning_filter_duration", _WARNING_FILTER_DURATION_FAIL)
        _run_category(watcher, config_path, "fail", fail_video,
                      _EXPECT_WARNING_FAIL, results)
    finally:
        watcher.destroy_node()
        rclpy.shutdown()
    update_json(config_path, "warning_filter_duration", _WARNING_FILTER_DURATION_SUCCESS)

    # ---------------------------- Summary ----------------------------
    print("=" * 80)
    print("Summary:")
    for label, name, found, expected, passed in results:
        verdict = "PASS" if passed else "FAIL"
        print(f"  [{verdict}] ({label:<7}) {name}  found={found!s:<5} expected={expected}")

    total = len(results)
    num_pass = sum(1 for *_, passed in results if passed)
    num_fail = total - num_pass
    print(f"\n{num_pass}/{total} episodes matched expectation; {num_fail} mismatched.")

    if num_fail:
        print("Mismatched episodes:")
        for label, name, found, expected, passed in results:
            if not passed:
                print(f"  - ({label}) {name}: got {found}, expected {expected}")

    # Nonzero exit on any mismatch so this can gate CI / scripts.
    sys.exit(1 if num_fail else 0)

if __name__ == "__main__":
    main(tyro.cli(Config))