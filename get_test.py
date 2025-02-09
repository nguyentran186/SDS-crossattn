import os

# Define folder path and output file
folder_path = "/root/SDS-crossattn/data/1/images"
output_file = "first_40_images.txt"

# Get list of image files sorted by name
valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
image_files = sorted(
    [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]
)

# Get first 40 images (or all if less than 40 exist)
first_40_images = image_files[:40]

# Write to output file
with open(output_file, "w") as f:
    for img in first_40_images:
        f.write(img + "\n")

print(f"First 40 images saved to {output_file}")
