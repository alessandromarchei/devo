import sys
import os
from tensorboard.backend.event_processing import event_accumulator

def main(event_file):
    if not os.path.exists(event_file):
        print(f"File not found: {event_file}")
        return

    print(f"Loading: {event_file}")
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()['scalars']
    if not tags:
        print("No scalar tags found in the file.")
        return

    print("Available scalar tags:")
    for tag in tags:
        print(f" - {tag}")

    for tag in tags:
        print(f"\n--- {tag} ---")
        events = ea.Scalars(tag)
        for e in events:
            print(f"Step: {e.step}, Value: {e.value}, Time: {e.wall_time}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_tfevents.py path/to/events.out.tfevents.*")
    else:
        main(sys.argv[1])
