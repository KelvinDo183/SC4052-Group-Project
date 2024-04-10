# SC4052-Group-Project


## Set up waifuwu with local comfy

0. install requirements.txt. Set up cuda according to local machine
```
pip install -r requirements.txt
```
1. Install checkpoints in 'inferences/models/checkpoints/'

    !!!Check for correct checkpoints path in `inference.py:Line 44`

    Current checkpoints:
    + anime: https://civitai.com/models/269232/aam-xl-anime-mix
    + realistic: https://civitai.com/models/139562?modelVersionId=361593
    + cartoon: https://civitai.com/models/81270

2. Get in 'inferences' directory (IMPORTANT)
```
cd inferences
```
3. Run inference server:
```
python sever.py
```

4. Run streamlit app;
```
streamlit run app.py
```

