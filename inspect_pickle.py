import pickle
import argparse

# Parse command-line argument
parser = argparse.ArgumentParser(description="Investigate a Pickle File")
parser.add_argument("pickle_file", type=str, help="Path to the pickle file")
args = parser.parse_args()

# Load the pickle file
with open(args.pickle_file, "rb") as f:
    data = pickle.load(f)

# Print general structure
print(f"\nPickle File Type: {type(data)}")
print(f"Number of Top-Level Keys: {len(data)}")

# Print first few keys and their types
sample_keys = list(data.keys())[:5]
print("\nSample Keys:")
for key in sample_keys:
    print(f"- {key} ({type(data[key])})")

# Investigate the first key in detail
first_key = sample_keys[0]
first_value = data[first_key]

print(f"\nStructure of First Key ({first_key}):")
if isinstance(first_value, dict):
    print(f" - Number of Subkeys: {len(first_value)}")
    for subkey, subvalue in first_value.items():
        print(f"   - {subkey}: {type(subvalue)}")
        if isinstance(subvalue, list) and len(subvalue) > 0:
            print(f"     - First 3 Elements: {subvalue[:3]}")
        elif isinstance(subvalue, dict):
            print(f"     - Subkeys: {list(subvalue.keys())[:3]}")
        else:
            print(f"     - Value: {subvalue}")

# Display additional sample entries
print("\nAdditional Sample Entries:")
for key in sample_keys[1:3]:  # Print details of two more entries
    print(f"\nKey: {key} ({type(data[key])})")
    value = data[key]
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            print(f"  - {subkey}: {type(subvalue)}")
            if isinstance(subvalue, list) and len(subvalue) > 0:
                print(f"    - First 3 Elements: {subvalue[:3]}")

# Count unique data types in the dictionary values
unique_types = set(type(value) for value in data.values())
print(f"\nUnique Types in Data Values: {unique_types}")

