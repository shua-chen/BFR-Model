import os
import torch
import contextlib
import time
import cv2
import numpy as np
from PIL import Image
import argparse
from safetensors.torch import load_file
import torch.nn as nn

from models.unet_max import UNet2DConditionModel
from models.controlnet_pdm import ControlNetModel
from pipeline.pipeline_controlnext import StableDiffusionXLControlNeXtPipeline
from diffusers import UniPCMultistepScheduler, AutoencoderKL,DiffusionPipeline
from transformers import AutoTokenizer, PretrainedConfig

def log_validation(
    vae, 
    unet, 
    controlnext, 
    args, 
    device='cuda',
    use_refiner=False,
):

    pipeline = StableDiffusionXLControlNeXtPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnext,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config()
    if args.lora_path is not None:
        pipeline.load_lora_weights(args.lora_path)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if use_refiner:
        refiner=DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        refiner = refiner.to(device)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    if args.negative_prompt is not None:
        negative_prompts = args.negative_prompt
        assert len(validation_prompts) == len(validation_prompts)
    else:
        negative_prompts = None

    image_logs = []
    inference_ctx = torch.autocast(device)

    for i, (validation_prompt, validation_image) in enumerate(zip(validation_prompts, validation_images)):
        ori_validation_image = Image.open(validation_image).convert("RGB")
        ori_validation_image = ori_validation_image.resize((args.resolution, args.resolution))
        validation_image=np.array(ori_validation_image)
        val_image_mean=validation_image.mean() / validation_image.mean(axis=(0,1))
        validation_image=(validation_image *val_image_mean).clip(0,255).astype(np.uint8)
        validation_image = Image.fromarray(validation_image).convert("RGB")
        images = []
        negative_prompt = negative_prompts[i] if negative_prompts is not None else None

        for _ in range(args.num_validation_images):
            with inference_ctx:
                if use_refiner:
                    image = pipeline(
                    prompt=validation_prompt, 
                    controlnet_image=validation_image, 
                    num_inference_steps=30, 
                    generator=generator, 
                    controlnet_scale=args.controlnext_scale,
                    negative_prompt=negative_prompt,
                    output_type="latent",
                ).images[0]
                    image = refiner(prompt=validation_prompt, 
                                    num_inference_steps=30, 
                                    image=image,
                                    height=args.resolution,
                                    width=args.resolution).images[0]
                else:
                    image = pipeline(
                    prompt=validation_prompt, 
                    controlnet_image=validation_image, 
                    num_inference_steps=30, 
                    generator=generator, 
                    controlnet_scale=args.controlnext_scale,
                    negative_prompt=negative_prompt,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            
            # image=np.array(image)
            # image=(image/val_image_mean).clip(0,255).astype(np.uint8)
            # image = Image.fromarray(image).convert("RGB")
            images.append(image)

        image_logs.append(
            {"validation_image": ori_validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        validation_image = log["validation_image"]

        formatted_images = []
        formatted_images.append(np.asarray(validation_image))
        for image in images:
            formatted_images.append(np.asarray(image))
        formatted_images = np.concatenate(formatted_images, 1)
        #formatted_images=formatted_images[1]

        file_path = os.path.join(save_dir_path, "{}.png".format(time.time()))
        formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_BGR2RGB)
        print("Save images to:", file_path)
        cv2.imwrite(file_path, formatted_images)

    return image_logs
    


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNeXt training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vae model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnext model or model identifier from huggingface.co/models."
        " If not specified controlnext weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or subset"
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae model or subset"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to lora"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnext_scale",
        type=float,
        default=1.0,
        help="Control level for the controlnext",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnext-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--save_load_weights_increaments",
        action="store_true",
        help=(
            "whether to store the weights_increaments"
        ),
    )
    parser.add_argument(
        "--use_refiner",
        action="store_true",
        help=(
            "whether to use refiner"
        ),
    )

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_name_or_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision if args.pretrained_vae_model_name_or_path is None else None,
        variant=args.variant if args.pretrained_vae_model_name_or_path is None else None,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )


    controlnext = ControlNetModel()
    if args.controlnet_model_name_or_path is not None:
        load_safetensors(controlnext, args.controlnet_model_name_or_path)
    else:
        controlnext.scale = 0.


    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        variant=args.variant
    )
    if args.unet_model_name_or_path is not None:
        load_safetensors(unet, args.unet_model_name_or_path, strict=False, load_weight_increasement=args.save_load_weights_increaments)
    if args.vae_model_name_or_path is not None:
        load_safetensors(vae,args.vae_model_name_or_path, strict=False)

    vae.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    controlnext.to(device)
    unet.to(device)

    log_validation(
        vae=vae, 
        unet=unet, 
        controlnext=controlnext, 
        args=args, 
        device=device,
        use_refiner=args.use_refiner,
    )