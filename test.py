import cv2
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline

# Specify the file paths.
image_path = "/root/SDS-crossattn/data/1/images/20220819_104221.jpg"
mask_path = "/root/SDS-crossattn/data/1/label/20220819_104221.jpg"

# Read the image using OpenCV.
# OpenCV loads images in BGR order by default.
image_cv = cv2.imread(image_path)
if image_cv is None:
    raise ValueError(f"Image not found at {image_path}")
# Convert the image from BGR to RGB.
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
# Convert to PIL Image.
input_image = Image.fromarray(image_rgb)

# Read the mask using OpenCV.
# For the mask, we read it in grayscale (single channel).
mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) * 255
if mask_cv is None:
    raise ValueError(f"Mask not found at {mask_path}")
# Convert to PIL Image (mode "L" for grayscale).
mask_image = Image.fromarray(mask_cv)

# Load the Stable Diffusion XL Inpainting pipeline.
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,  # Use float16 for efficiency.
).to("cuda")  # Move model to GPU.

prompt = ("enhance the quality")

# Run the inpainting pipeline.
result = pipe(prompt=prompt, image=input_image, mask_image=mask_image, num_inference_steps=20)

# Get the output image and save it.
output_image = result.images[0]
output_image.save("inpainted_output.png")

