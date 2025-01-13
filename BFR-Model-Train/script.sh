python run_controlnext.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
 --variant fp16 \
 --output_dir="test" \
 --validation_image "/scratch/students/2024-fall-shuhua/mydataset/test_set/1024/0008_lq.png" \
 --validation_prompt "a high-quality, natural-look image of a human face"  \
 --controlnet_model_name_or_path "pretrained_weights/controlnet.safetensors" \
 --unet_model_name_or_path "pretrained_weights/unet_weight_increasements.safetensors" \
 --num_validation_images 1 \
 --controlnext_scale 1 \
 --resolution 1024 


