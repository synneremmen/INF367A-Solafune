import os
import shutil
import subprocess

source_dir_masked = "data/masked_annotations/"
dest_dir_masked = "data/masked_annotations_subset/"

source_dir_train = "data/train_images/"
dest_dir_train = "data/train_images_subset/"

if not os.path.exists(source_dir_train):
    raise FileNotFoundError(f"Folder {source_dir_train} not found.")

elif not os.path.exists(source_dir_masked):
    subprocess.run(["python", "create_masked_data.py"], check=True)

else:

    for source_dir, dest_dir in zip([source_dir_masked, source_dir_train], [dest_dir_masked, dest_dir_train]):
        # List of files to copy
        subset = [f"train_{i}.tif" for i in range(11)]

        os.makedirs(dest_dir, exist_ok=True)

        for file in os.listdir(source_dir):
            if file in subset:
                file_path = os.path.join(source_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy(file_path, dest_path)
                print(f"Copied {file} to {dest_dir}")

        print("Subset generated.")
