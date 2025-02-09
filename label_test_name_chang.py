import os
import shutil

# Define paths
folder1 = "/root/SDS-crossattn/data/test_label"  # Original images
new_folder = "/root/SDS-crossattn/data/test_label_renamed"  # Output folder
txt_file = "/root/SDS-crossattn/data/1/sparse/0/test.txt"  # Text file with new names

# Ensure output folder exists
os.makedirs(new_folder, exist_ok=True)

# Read new names from text file
with open(txt_file, "r") as f:
    new_names = [line.strip() for line in f.readlines()]  # Remove whitespace

# Get sorted list of images
images1 = sorted(os.listdir(folder1))  # Images to be renamed

# Ensure equal number of images and names
if len(images1) != len(new_names):
    print("Error: Number of images and names do not match!")
else:
    for img1, new_name in zip(images1, new_names):
        src_path = os.path.join(folder1, img1)
        dest_path = os.path.join(new_folder, new_name)  # Rename using txt file
        shutil.copy(src_path, dest_path)  # Copy with new name

    print(f"Renamed {len(images1)} images and saved to {new_folder}")
