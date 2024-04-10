# Wifu Creator
## Ready to design your dream waifu? Dive into our Waifu Creator! This intuitive app empowers you to craft your unique waifu with endless possibilities. Start creating now!
<img width="1180" alt="methodology" src="https://th.bing.com/th/id/OIP.PgYeWuom0-xflkachWRk0gHaEK?rs=1&pid=ImgDetMain">

## Our app offers a wide array of model options, providing users with numerous possibilities to craft their ideal waifu. Dive into the creative process with endless variations at your fingertips. Follow the instructions below to embark on your world of wifu!
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

