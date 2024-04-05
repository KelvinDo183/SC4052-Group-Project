import time
from ast import literal_eval
from io import BytesIO

import streamlit as st
import torch
from database import DatabaseManager
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler


st.set_page_config(layout="wide") 
db_manager = DatabaseManager()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def generate_image(model_name, model_kwargs):
    t0 = time.time()
    if model_name == "Stable Diffusion":    # like shit
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if model_name == "SDXL":                # better
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", use_karras_sigmas=True, euler_at_final=True).to(device)
        pipe.enable_xformers_memory_efficient_attention()
    image = pipe(**model_kwargs).images[0]
    print("Taken:", time.time()-t0)
    return image


def login():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        authenticated, user_id, username = db_manager.authenticate(username, password)
        if authenticated:
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.rerun()
            return True
        else:
            st.error('Invalid username or password. Please try again.')
    return False


def create_account_page():
    st.title('Create Account')
    new_username = st.text_input('New Username')
    new_password = st.text_input('New Password', type='password')
    if st.button('Create Account'):
        if new_username.strip() and new_password.strip():
            db_manager.create_account(new_username, new_password)
            st.success('Account created successfully!')
            st.write('You can now login with your new account.')
        else:
            st.error('Username and password cannot be empty.')


def profile_page():
    user_id = st.session_state.user_id
    user_images = db_manager.get_user_images(user_id)
    st.header(f"{st.session_state.username} Profile")

    if len(user_images) == 0:
        st.write("Welcome new user, you don't have any generated images. Please go cook your new waifu~")
    else:
        num_columns = 5
        images_to_delete = []
        images_per_row = len(user_images) // num_columns + 1
        for i in range(images_per_row):
            col = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < len(user_images):
                    image = user_images[index]
                    delete_checkbox = col[j].checkbox(f"Delete Image {image[0]}", key=f"delete_{image[0]}")
                    if delete_checkbox:
                        images_to_delete.append(image[0])
                    col[j].image(image[-1], caption=f"Model: {image[2]}, Prompt: {literal_eval(image[3])['prompt']}")

        if st.button("Delete Images"):
            if images_to_delete:
                db_manager.delete_images(images_to_delete)
                st.experimental_rerun()


def generate_page():
    st.subheader('Generate and Save Image 👨‍🎨')
    model_name = st.selectbox('Model', ['Stable Diffusion', 'SDXL'])
    model_kwargs = {
        "prompt": st.text_input('Prompt', placeholder="An astronaut riding a rainbow unicorn, cinematic, dramatic")
    }

    with st.expander("Refine your output"):
        model_kwargs["negative_prompt"] = st.text_input('Negative Prompt', placeholder="the absolute worst quality, distorted features",
                                                        help="Basically type what you don't want to see in the generated image")
        col1, col2 = st.columns(2)
        with col1:
            model_kwargs['width'] = st.number_input("Width of output image", value=1024)
        with col2:
            model_kwargs['height'] = st.number_input("Height of output image", value=1024)
        col1, col2 = st.columns(2)
        with col1:
            model_kwargs['num_inference_steps'] = st.slider("Number of denoising steps", value=20, min_value=1, max_value=100, step=1,
                                                            help="More steps result in higher quality but also require more time to generate.")
        with col2:
            model_kwargs['guidance_scale'] = st.slider("Guidance scale", value=7.5, min_value=1.0, max_value=20.0, step=0.1,
                                                            help="Determines how similar the image will be to the prompt. Note that higher value results in less 'creative' image.")

    if st.button('Generate and Save'):
        if model_kwargs["prompt"]:
            with st.spinner("Generating image..."):
                generated_image = generate_image(model_name, model_kwargs)
            st.success('Done!')
            if generated_image:
                st.image(generated_image, caption='Generated Image',
                        use_column_width=True)
                filename = f"{st.session_state.username}_generated_image.png"
                generated_image.save(filename)
                st.success(f"Image saved as {filename}")
                img_data = BytesIO()
                generated_image.save(img_data, format='PNG')
                img_data.seek(0)
                db_manager.insert_image(st.session_state.user_id, model_name, model_kwargs, img_data.read())
        else:
            st.warning("Please enter a prompt.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.sidebar.write(f'Welcome, {st.session_state.username}! ✨')
        page = st.sidebar.radio("What would you like to do today?", ["View Profile", "Generate Image"])
        if page == "View Profile":
            profile_page()
        else:
            generate_page()
    else:
        page = st.sidebar.radio("Navigation", ["Login", "Create Account"])
        if page == "Login":
            login()
        else:
            create_account_page()


if __name__ == '__main__':
    main()
