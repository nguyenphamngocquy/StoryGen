import os
from typing import Optional
from torchvision import transforms
from PIL import Image
from typing import List, Optional, Union

import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint
import argparse

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
    prompt: str,
    ref_prompt: Union[str, List[str]],
    ref_image: Union[str, List[str]],
    logdir: str = "./inference_StorySalon/",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 3.5,
    num_sample_per_prompt: int = 10,
    stage: str = "multi-image-condition", # ["multi-image-condition", "auto-regressive", "no"]
    mixed_precision: Optional[str] = "fp16" ,
):
    time_string = get_time_string()
    logdir += f"_{time_string}"
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    # pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_path, local_files_only=True)
    # text_encoder = pipeline.text_encoder
    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
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
        r_image = Image.open(id).convert('RGB').resize((512, 512))
        r_image = transforms.ToTensor()(r_image)
        ref_images.append(np.ascontiguousarray(r_image))
    ref_images = torch.from_numpy(np.ascontiguousarray(ref_images)).float()
    for ref_image in ref_images:
        ref_image = ref_image * 2. - 1.
    ref_images = ref_images.unsqueeze(0)

    sample_seeds = torch.randint(0, 100000, (num_sample_per_prompt,))
    sample_seeds = sorted(sample_seeds.numpy().tolist())
    
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
            height = 512,
            width = 512,
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

    parser = argparse.ArgumentParser(description="Run inference with parameters")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--logdir", type=str, default=None, help="Path to save inference results")
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    parser.add_argument("--prev_p", nargs="+", required=True, help="Previous prompts (space-separated list)")
    parser.add_argument("--ref_image", nargs="+", required=True, help="Reference image paths (space-separated list)")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale for text")
    parser.add_argument("--image_guidance_scale", type=float, default=None, help="Guidance scale for images")
    parser.add_argument("--num_sample_per_prompt", type=int, default=None, help="Number of samples per prompt")
    parser.add_argument("--stage", type=str, choices=["multi-image-condition", "auto-regressive", "no"], default=None, help="Inference stage")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default=None, help="Mixed precision mode")
    args = parser.parse_args()

    test(
        pretrained_model_path=args.pretrained_model_path,
        prompt=args.prompt,
        ref_prompt=args.prev_p,
        ref_image=args.ref_image,
        logdir=args.logdir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_sample_per_prompt=args.num_sample_per_prompt,
        stage=args.stage,
        mixed_precision=args.mixed_precision,
    )

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py