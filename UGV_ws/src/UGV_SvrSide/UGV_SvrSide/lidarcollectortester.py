#!/usr/bin/env python3
"""
bag_integrity_checker.py

Checks the integrity and publish rate of a ROS 2 bag containing
Livox LiDAR (/livox/lidar) and IMU (/livox/imu) data.

Usage:
    python3 bag_integrity_checker.py <path_to_bag_directory>

Output:
    - Per-topic message count
    - Per-topic publish rate (mean, min, max, std)
    - Gap detection (intervals exceeding 3x the expected period)
    - Timestamp monotonicity check
    - Summary pass/fail for each topic
"""

import sys
import os
import numpy as np
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


# ── Configuration ─────────────────────────────────────────────────────────────

TOPICS = {
    '/livox/lidar': {
        'expected_rate_hz': 10.0,          # Livox Avia default publish rate
        'type': 'livox_interfaces/msg/CustomMsg',
    },
    '/livox/imu': {
        'expected_rate_hz': 200.0,         # Livox Avia IMU default rate
        'type': 'sensor_msgs/msg/Imu',
    },
}

# A gap is flagged if the interval between two consecutive messages
# exceeds this multiplier times the expected period.
GAP_THRESHOLD_MULTIPLIER = 3.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_reader(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def collect_timestamps(bag_path: str, topics: dict) -> dict:
    """
    Reads the bag and collects header timestamps (in seconds) for each topic.
    Falls back to bag receive timestamps if the message has no header.
    """
    reader = open_reader(bag_path)

    # Filter to only the topics we care about
    filter_ = rosbag2_py.StorageFilter(topics=list(topics.keys()))
    reader.set_filter(filter_)

    # Build a type map so we can deserialise
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    data = {topic: {'timestamps': [], 'recv_timestamps': []} for topic in topics}

    while reader.has_next():
        topic, raw, recv_ts = reader.read_next()
        if topic not in topics:
            continue

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(raw, msg_type)

        # Use header stamp if available, else fall back to receive timestamp
        if hasattr(msg, 'header'):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            t = recv_ts * 1e-9  # nanoseconds → seconds

        data[topic]['timestamps'].append(t)
        data[topic]['recv_timestamps'].append(recv_ts * 1e-9)

    return data


def analyse_topic(topic: str, cfg: dict, timestamps: list) -> dict:
    """
    Computes rate statistics and detects gaps/non-monotonicity for one topic.
    """
    result = {
        'topic': topic,
        'expected_rate_hz': cfg['expected_rate_hz'],
        'message_count': len(timestamps),
        'duration_s': 0.0,
        'mean_rate_hz': 0.0,
        'mean_interval_ms': 0.0,
        'min_interval_ms': 0.0,
        'max_interval_ms': 0.0,
        'std_interval_ms': 0.0,
        'gaps': [],
        'non_monotonic_count': 0,
        'passed': False,
    }

    if len(timestamps) < 2:
        print(f"  [WARN] {topic}: fewer than 2 messages — cannot compute rates.")
        return result

    ts = np.array(timestamps)

    # Monotonicity check
    diffs = np.diff(ts)
    non_mono = int(np.sum(diffs <= 0))
    result['non_monotonic_count'] = non_mono

    # Remove non-positive intervals for rate stats (keep meaningful intervals)
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        print(f"  [ERROR] {topic}: all intervals are zero or negative.")
        return result

    intervals_ms = positive_diffs * 1000.0
    result['duration_s']       = float(ts[-1] - ts[0])
    result['mean_rate_hz']     = float(1.0 / np.mean(positive_diffs))
    result['mean_interval_ms'] = float(np.mean(intervals_ms))
    result['min_interval_ms']  = float(np.min(intervals_ms))
    result['max_interval_ms']  = float(np.max(intervals_ms))
    result['std_interval_ms']  = float(np.std(intervals_ms))

    # Gap detection
    expected_period_s = 1.0 / cfg['expected_rate_hz']
    gap_threshold_s   = GAP_THRESHOLD_MULTIPLIER * expected_period_s
    gap_indices       = np.where(positive_diffs > gap_threshold_s)[0]

    for idx in gap_indices:
        result['gaps'].append({
            'at_s':     float(ts[idx]),
            'gap_ms':   float(positive_diffs[idx] * 1000.0),
        })

    # Pass criteria:
    #   - rate within ±20 % of expected
    #   - no non-monotonic timestamps
    #   - no gaps
    rate_ok  = abs(result['mean_rate_hz'] - cfg['expected_rate_hz']) / cfg['expected_rate_hz'] < 0.20
    mono_ok  = non_mono == 0
    gaps_ok  = len(result['gaps']) == 0
    result['passed'] = rate_ok and mono_ok and gaps_ok

    return result


def print_report(results: list):
    sep = '─' * 70
    print()
    print('=' * 70)
    print('  ROS 2 BAG INTEGRITY REPORT')
    print('=' * 70)

    all_passed = True

    for r in results:
        print()
        print(sep)
        print(f"  Topic : {r['topic']}")
        print(sep)
        print(f"  Messages          : {r['message_count']}")
        print(f"  Duration          : {r['duration_s']:.2f} s")
        print(f"  Expected rate     : {r['expected_rate_hz']:.1f} Hz")
        print(f"  Mean rate         : {r['mean_rate_hz']:.2f} Hz")
        print(f"  Mean interval     : {r['mean_interval_ms']:.2f} ms")
        print(f"  Min  interval     : {r['min_interval_ms']:.2f} ms")
        print(f"  Max  interval     : {r['max_interval_ms']:.2f} ms")
        print(f"  Std  interval     : {r['std_interval_ms']:.2f} ms")
        print(f"  Non-monotonic ts  : {r['non_monotonic_count']}")

        if r['gaps']:
            print(f"  Gaps (>{GAP_THRESHOLD_MULTIPLIER}x period) : {len(r['gaps'])}")
            # for g in r['gaps']:
            #     # print(f"      t={g['at_s']:.3f}s  gap={g['gap_ms']:.1f}ms")
        else:
            print(f"  Gaps              : 0")

        status = "PASS" if r['passed'] else "FAIL"
        print(f"  Result            : {status}")
        if not r['passed']:
            all_passed = False

    print()
    print('=' * 70)
    overall = "PASS" if all_passed else "FAIL"
    print(f"  OVERALL RESULT: {overall}")
    print('=' * 70)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bag_integrity_checker.py <path_to_bag_directory>")
        sys.exit(1)

    bag_path = sys.argv[1]

    if not os.path.isdir(bag_path):
        print(f"[ERROR] Bag directory not found: {bag_path}")
        sys.exit(1)

    print(f"[INFO] Analysing bag: {bag_path}")
    print(f"[INFO] Topics checked: {list(TOPICS.keys())}")

    rclpy.init()

    try:
        raw_data = collect_timestamps(bag_path, TOPICS)
    finally:
        rclpy.shutdown()

    results = []
    for topic, cfg in TOPICS.items():
        print(f"\n[INFO] Processing {topic} ...")
        ts = raw_data[topic]['timestamps']
        print(f"       {len(ts)} messages found")
        r = analyse_topic(topic, cfg, ts)
        results.append(r)

    print_report(results)


if __name__ == '__main__':
    main()