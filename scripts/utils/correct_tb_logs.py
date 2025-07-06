import os
import sys
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict

def load_all_scalars_by_tag(folder):
    tag_data = defaultdict(list)

    for fname in sorted(os.listdir(folder)):
        if not fname.startswith("events.out.tfevents"):
            continue
        path = os.path.join(folder, fname)
        for e in summary_iterator(path):
            for v in e.summary.value:
                tag_data[v.tag].append({
                    "step": e.step,
                    "wall_time": e.wall_time,
                    "value": v.simple_value
                })
    return tag_data

def detect_jump(steps, tolerance=2.0):
    if len(steps) < 3:
        return None, 0

    diffs = [steps[i+1] - steps[i] for i in range(min(5, len(steps)-1))]
    expected_delta = min(diffs)

    for i in range(1, len(steps)):
        delta = steps[i] - steps[i-1]
        if abs(delta - expected_delta) > expected_delta * tolerance:
            jump_step = steps[i]
            correction = jump_step - (steps[i-1] + expected_delta)
            return jump_step, correction

    return None, 0

def correct_tag_data(tag_data):
    corrected = {}
    for tag, entries in tag_data.items():
        sorted_entries = sorted(entries, key=lambda x: x['step'])
        steps = [e['step'] for e in sorted_entries]
        jump_step, correction = detect_jump(steps)

        if correction == 0:
            corrected[tag] = sorted_entries
        else:
            print(f"🛠 Fixing jump for tag '{tag}': step {jump_step} corrected by −{correction}")
            for e in sorted_entries:
                if e['step'] >= jump_step:
                    e['step'] -= correction
            corrected[tag] = sorted_entries

    return corrected

def overwrite_specific_avg_values(tag_data, overwrite_dict):
    """Modifies AVG tag: overwrite_dict = {step: new_value}"""
    if "AVG" not in tag_data:
        print("⚠️ AVG tag not found. Skipping manual overwrite.")
        return tag_data

    modified = 0
    for e in tag_data["AVG"]:
        if e["step"] in overwrite_dict:
            e["value"] = overwrite_dict[e["step"]]
            modified += 1
    print(f"✏️ Overwritten {modified} AVG values manually.")
    return tag_data

def write_merged_event_file(corrected_data, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for tag, entries in corrected_data.items():
            for e in entries:
                event = tf.compat.v1.Event(
                    wall_time=e['wall_time'],
                    step=e['step'],
                    summary=tf.compat.v1.Summary(value=[
                        tf.compat.v1.Summary.Value(tag=tag, simple_value=e['value'])
                    ])
                )
                writer.write(event.SerializeToString())
    print(f"✅ Saved corrected log to: {output_path}")

def merge_and_fix_all(folder, output_file="events.merged.fixed.tfevents"):
    print(f"📂 Processing folder: {folder}")
    tag_data = load_all_scalars_by_tag(folder)
    print(f"📊 Found {len(tag_data)} tags")

    # === Correct step jumps
    corrected_data = correct_tag_data(tag_data)

    # === Manual overwrite for AVG
    manual_avg_overwrites = {
        20000: 0.28,
        25000: 1.30,
        30000: 0.47,
        35000: 0.59,
        40000: 0.76,
        45000: 0.37,
        50000: 0.83,
        55000: 0.59,
        60000: 0.24,
        65000: 0.41,
        70000: 0.34,
        75000: 0.25,
        80000: 0.38,
        85000: 0.23,
        90000: 0.29,
        95000: 0.18,
        100000: 0.25,
        105000: 0.20,
        110000: 0.28,
        115000: 0.20,
        120000: 0.16
    }
    corrected_data = overwrite_specific_avg_values(corrected_data, manual_avg_overwrites)

    # === Delete previous event files inside the folder
    for fname in os.listdir(folder):
        if fname.startswith("events.out.tfevents"):
            os.remove(os.path.join(folder, fname))
            print(f"🗑️ Deleted old event file: {fname}")
    # === Save output
    output_path = os.path.join(folder, output_file)
    write_merged_event_file(corrected_data, output_path)

# === Entry point ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_and_fix_tb_events.py <folder_with_event_files>")
        sys.exit(1)

    folder = sys.argv[1]
    merge_and_fix_all(folder)
