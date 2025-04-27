import os
import shutil
import subprocess
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Scripts to create subsets of the dataset
# -------------------------------------------------------------------

load_dotenv()
IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("IMAGES_PATH")) 
MASKED_IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("MASKED_IMAGES_PATH")) 
IMAGES_SUBSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("IMAGES_SUBSET_PATH")) 
MASKED_IMAGES_SUBSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("MASKED_IMAGES_SUBSET_PATH")) 

source_dir_masked = MASKED_IMAGES_PATH
dest_dir_masked = MASKED_IMAGES_SUBSET_PATH
source_dir_train = IMAGES_PATH
dest_dir_train = IMAGES_SUBSET_PATH

if not os.path.exists(source_dir_train):
    raise FileNotFoundError(f"Folder {source_dir_train} not found.")

elif not os.path.exists(source_dir_masked):
    subprocess.run(["python", "utils/create_masked_data.py"], check=True)

else:

    for source_dir, dest_dir in zip([source_dir_masked, source_dir_train], [dest_dir_masked, dest_dir_train]):
        # List of files to copy
        subset = [f"train_{i}.tif" for i in range(30)]

        os.makedirs(dest_dir, exist_ok=True)

        for file in os.listdir(source_dir):
            if file in subset:
                file_path = os.path.join(source_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy(file_path, dest_path)
                print(f"Copied {file} to {dest_dir}")

        print("Subset generated.")
