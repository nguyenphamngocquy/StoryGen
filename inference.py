import os
from typing import Optional
from torchvision import transforms
from PIL import Image
from typing import List, Optional, Union

import argparse
import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import get_time_string
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline

logger = get_logger(__name__)

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    ref_prompt: Union[str, List[str]],
    ref_image: Union[str, List[str]],
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 3.5,
    num_sample_per_prompt: int = 10,
    stage: str = "multi-image-condition", # ["multi-image-condition", "auto-regressive", "no"]
    mixed_precision: Optional[str] = "fp16" ,
    height: int = 512,
    width: int = 512,
):
    time_string = get_time_string()
    logdir = os.path.join(logdir, f"_{time_string}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")

    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    ref_images= []
    for id in ref_image:
        r_image = Image.open(id).convert('RGB').resize((height, width))
        r_image = transforms.ToTensor()(r_image)
        ref_images.append(np.ascontiguousarray(r_image))
    ref_images = torch.from_numpy(np.ascontiguousarray(ref_images)).float()
    for ref_image in ref_images:
        ref_image = ref_image * 2. - 1.
    ref_images = ref_images.unsqueeze(0)

    sample_seeds = torch.randint(0, 100000, (num_sample_per_prompt,))
    sample_seeds = sorted(sample_seeds.numpy().tolist())

    if accelerator.is_main_process:
        print("ref_images: ", ref_images.shape)
        print("sample_seeds: ", sample_seeds)

    generator = []
    for seed in sample_seeds:
        generator_temp = torch.Generator(device=accelerator.device)
        generator_temp.manual_seed(seed)
        generator.append(generator_temp)
    with torch.no_grad():
        output = pipeline(
            stage = stage,
            prompt = prompt,
            image_prompt = ref_images,
            prev_prompt = ref_prompt,
            height = height,
            width = width,
            generator = generator,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            image_guidance_scale = image_guidance_scale,
            num_images_per_prompt=num_sample_per_prompt,
        ).images
    
    images = []
    for i, image in enumerate(output):
        images.append(image[0])
        images[i].save(os.path.join(logdir, f"{sample_seeds[i]}_output.png"))


if __name__ == "__main__":

    # pretrained_model_path = '/checkpoint_StorySalon/'
    # logdir = "./inference_StorySalon/"
    # num_inference_steps = 40
    # guidance_scale = 7
    # image_guidance_scale = 3.5
    # num_sample_per_prompt = 10
    # mixed_precision = "fp16"
    # stage = 'auto-regressive' # ["multi-image-condition", "auto-regressive", "no"]
    
    # prompt = "The white cat is running after the black-haired man."
    # prev_p = ["The black-haired man", "The white cat."]
    # ref_image = ["./data/boy.jpg", 
    #              ".data/whitecat1.jpg"]

    # test(pretrained_model_path, 
    #      logdir, 
    #      prompt, 
    #      prev_p, 
    #      ref_image, 
    #      num_inference_steps, 
    #      guidance_scale, 
    #      image_guidance_scale, 
    #      num_sample_per_prompt,
    #      stage, 
    #      mixed_precision)

    parser = argparse.ArgumentParser(description="Run the inference test.")

    # Add arguments
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model.")
    parser.add_argument("--logdir", type=str, help="Logging directory.")
    parser.add_argument("--prompt", type=str, help="Prompt for text-to-image generation.")
    parser.add_argument("--ref_prompt", type=str, nargs="+", help="Reference text prompts (space-separated list).")
    parser.add_argument("--ref_image", type=str, nargs="+", help="Reference image paths (space-separated list).")
    parser.add_argument("--num_inference_steps", type=int, help="Number of inference steps for validation.")
    parser.add_argument("--guidance_scale", type=float, help="Guidance scale for validation.")
    parser.add_argument("--image_guidance_scale", type=float, help="Image guidance scale for conditioning.")
    parser.add_argument("--num_sample_per_prompt", type=int, help="Number of samples per prompt.")
    parser.add_argument("--stage", type=str, choices=["multi-image-condition", "auto-regressive", "no"], help="Generation stage mode.")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "no"], help="Mixed precision mode.")
    parser.add_argument("--height", type=int, help="Height of output image (must be divisible by 8).")
    parser.add_argument("--width", type=int, help="Width of output image (must be divisible by 8).")

    args = parser.parse_args()
    params = {k: v for k, v in vars(args).items() if v is not None}

    # Convert single-item lists to strings
    for key in ["ref_prompt", "ref_image"]:
        if key in params and isinstance(params[key], list) and len(params[key]) == 1:
            params[key] = params[key][0]

    test(**params)

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py