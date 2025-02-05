import os
import shutil

source_dir = "data/masked_annotations/"
dest_dir = "data/masked_annotations_subset/"

# List of files to copy
subset = [
    "train_0.tif", "train_1.tif", "train_2.tif" "train_3.tif", "train_4.tif", 
    "train_5.tif", "train_6.tif", "train_7.tif", "train_8.tif", 
    "train_9.tif", "train_10.tif"
]

os.makedirs(dest_dir, exist_ok=True)

for file in os.listdir(source_dir):
    if file in subset:
        file_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(file_path, dest_path)
        print(f"Copied {file} to {dest_dir}")

print("File copying completed.")
