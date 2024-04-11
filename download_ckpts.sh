cd inferences/models
mkdir loras
mkdir checkpoints

cd loras
wget "https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_8step_lora.safetensors" -O "sdxl_lightning_8step_lora.safetensors"

cd ../checkpoints
wget "https://civitai.com/api/download/models/128080" -O "sdXL_v10RefinerVAEFix.safetensors"
wget "https://civitai.com/api/download/models/303526?type=Model&format=SafeTensor&size=full&fp=fp16" -O "aamXLAnimeMix_v10.safetensors"
wget "https://civitai.com/api/download/models/144566" -O "samaritan3dCartoon_v40SDXL.safetensors"
wget "https://civitai.com/api/download/models/361593" -O "realvisxlV40_v40LightningBakedvae.safetensors"

