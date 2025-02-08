from guidance.sd_step import train_step
from PIL import Image
import torch

image = Image.open("/root/workspace/dataset/1/images/20220819_104221.jpg")
mask_image = Image.open("/root/workspace/dataset/1/images/20220819_104221.jpg")

loss = train_step("fuck", image, mask_image)
loss.backward()

# from diffusers import StableDiffusionXLInpaintPipeline
# import torch

# # Load the inpainting model
# pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
#     torch_dtype=torch.float16,  # Use float16 for efficiency
# ).to("cuda")  # Move model to GPU if available

# # Print a success message
# print("Stable Diffusion XL Inpainting model loaded successfully!")
# breakpoint()