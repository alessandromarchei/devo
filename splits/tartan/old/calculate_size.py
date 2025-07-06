total_size = 0

with open('/home/amarchei/Desktop/repos/DEVO/splits/tartan/events_size.txt', 'r') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            size, _ = line.split(maxsplit=1)
            size_in_mb = float(size.replace('MB', '').strip())
            total_size += size_in_mb

print(f"Total size of the dataset: {total_size:.2f} MB")
