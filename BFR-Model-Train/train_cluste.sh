accelerate launch train_controlnext.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--variant fp16 \
--use_safetensors \
--unet_trainable_param_pattern ".*attn2.*to_out.*" \
--output_dir "trains/cssft_pdm_19k/example" \
--logging_dir "logs" \
--learning_rate 5e-6 \
--lr_warmup_steps 300 \
--learning_rate_controlnet 5e-5 \
--resolution 1024 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--controlnet_scale_factor 1 \
--mixed_precision fp16 \
--enable_xformers_memory_efficient_attention \
--train_data_dir "/scratch/students/2024-fall-shuhua/mydataset/FFHQ/1024" \
--train_data_opt "/scratch/students/2024-fall-shuhua/code/SP-CP/data_opt.yaml" \
--image_column "image" \
--conditioning_image_column "depth_map" \
--caption_column "caption" \
--validation_image "examples/vidit_depth/condition_0_1024.png" \
--validation_prompt "a high-quality, natural-look image of a human face" \
--validation_steps 100 \
--num_train_epochs 20 \
--train_batch_size 12 \
#--report_to "wandb" 

