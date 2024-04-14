# Wifu Creator
## Ready to design your dream waifu? Dive into our Waifu Creator! This intuitive app empowers you to craft your unique waifu with endless possibilities. Start creating now!
<img width="1180" alt="methodology" src="https://th.bing.com/th/id/OIP.PgYeWuom0-xflkachWRk0gHaEK?rs=1&pid=ImgDetMain">

## Our app offers a wide array of model options, providing users with numerous possibilities to craft their ideal waifu. Dive into the creative process with endless variations at your fingertips. Follow the instructions below to embark on your world of wifu!
## Set up waifuwu with local comfy

0. install requirements.txt. Set up cuda according to local machine
```
conda create -n cloud python=3.11 -y
conda activate cloud
pip install -r requirements.txt
```
1. Install checkpoints in 'inferences/models/checkpoints/'

    Check for correct checkpoints path in [`inference.py`](inferences/inference.py#L43)

    Base LoRA: [sdxl_lightning_8step_lora.safetensors](https://huggingface.co/ByteDance/SDXL-Lightning/tree/main)
    SDXL Refiner: [sdXL_v10RefinerVAEFix.safetensors](https://civitai.com/models/101055?modelVersionId=128080&fbclid=IwAR3YkVt0HHZ5mwLHySiPp8S6PXsSORQ7NCdTk-pC4URXWmPmWNZPYmMciAU)
    Current checkpoints: [anime](https://civitai.com/models/269232/aam-xl-anime-mix), [realistic](https://civitai.com/models/139562?modelVersionId=361593),[cartoon](https://civitai.com/models/81270) 

```bash
# Or simply run this
bash download_ckpts.sh
```

2. Run inference server:
```bash
cd inferences   # must do
python sever.py
```

3. Open another terminal and run streamlit app;
```bash
streamlit run app.py
```

