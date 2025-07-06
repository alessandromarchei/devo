import os
import sys
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict

def load_events_from_file(file_path):
    for event in summary_iterator(file_path):
        yield event

def load_all_events(folder_or_file):
    if os.path.isfile(folder_or_file):
        return list(load_events_from_file(folder_or_file))

    all_events = []
    for fname in sorted(os.listdir(folder_or_file)):
        if fname.startswith("events.out.tfevents"):
            path = os.path.join(folder_or_file, fname)
            all_events.extend(load_events_from_file(path))
    return all_events

def cut_and_sample_events(events, max_step=120000, stride_loss_train=1):
    last_kept_loss_step = -stride_loss_train
    final_events = []

    for event in sorted(events, key=lambda e: e.wall_time):
        if event.step > max_step:
            continue

        new_values = []
        for v in event.summary.value:
            if v.tag == "loss/train":
                if event.step - last_kept_loss_step >= stride_loss_train or event.step == 0:
                    new_values.append(v)
                    last_kept_loss_step = event.step
            else:
                new_values.append(v)

        if new_values:
            new_event = tf.compat.v1.Event(
                wall_time=event.wall_time,
                step=event.step,
                summary=tf.compat.v1.Summary(value=new_values)
            )
            final_events.append(new_event)

    return final_events

def write_event_file(events, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for e in events:
            writer.write(e.SerializeToString())
    print(f"✅ Saved {len(events)} events to {output_path}")

# === CLI ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cut_and_sample_events.py <input_file_or_folder> [max_step] [stride_for_loss_train]")
        sys.exit(1)

    input_path = sys.argv[1]
    max_step = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    base = os.path.basename(input_path.rstrip("/"))
    out_suffix = f".cut_to_{max_step}_loss_train_stride_{stride}"
    output_file = os.path.join(
        os.path.dirname(input_path),
        f"{base}{out_suffix}.tfevents"
    )

    print(f"📥 Loading events from: {input_path}")
    events = load_all_events(input_path)

    print(f"✂️ Cutting at step {max_step}, applying stride={stride} to tag 'loss/train'...")
    cleaned = cut_and_sample_events(events, max_step=max_step, stride_loss_train=stride)

    write_event_file(cleaned, output_file)
