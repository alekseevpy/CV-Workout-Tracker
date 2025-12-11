# import torch
# from diffusers import DiffusionPipeline

# DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", dtype=torch.bfloat16, device_map="cuda")



import torch
from diffusers import StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", dtype=torch.bfloat16, device_map="cuda")