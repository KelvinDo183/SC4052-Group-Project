import streamlit as st
import sqlite3
import requests
from PIL import Image
from io import BytesIO
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


conn_users = sqlite3.connect('users.db')
conn_images = sqlite3.connect('images.db')
c_users = conn_users.cursor()
c_images = conn_images.cursor()


c_users.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)''')
conn_users.commit()


c_images.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, prompt TEXT, model TEXT, image BLOB)''')
conn_images.commit()


def authenticate(username, password):
    c_users.execute(
        "SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c_users.fetchone()
    if user:
        return True, user[0], user[1]
    else:
        return False, None, None


def create_account(username, password):
    c_users.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn_users.commit()


def insert_image(user_id, prompt, model, image):
    c_images.execute("INSERT INTO images (user_id, prompt, model, image) VALUES (?, ?, ?, ?)",
                     (user_id, prompt, model, image))
    conn_images.commit()


def get_images_by_user(user_id):
    c_images.execute("SELECT * FROM images WHERE user_id=?", (user_id,))
    return c_images.fetchall()


def generate_image(prompt, model):
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=True)
    with autocast("cuda"):
        output = pipe(prompt, guidance_scale=7.5)
    image = output["images"][0]
    image.save("astronaut-riding-horse.png")
    img = Image.open(BytesIO(image))
    return img


def login():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    login_button = st.button('Login')

    if login_button:
        authenticated, user_id, username = authenticate(username, password)
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
    create_button = st.button('Create Account')

    if create_button:
        if new_username.strip() and new_password.strip():
            create_account(new_username, new_password)
            st.success('Account created successfully!')
            st.write('You can now login with your new account.')
        else:
            st.error('Username and password cannot be empty.')


def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.write(f'Welcome, {st.session_state.username}!')
        st.subheader('Generate and Save Image')
        prompt = st.text_input('Prompt')
        model = st.selectbox(
            'Model', ['Stable Diffusion', 'Model 2', 'Model 3'])
        if st.button('Generate and Save'):
            if prompt:
                generated_image = generate_image(prompt, model)
                if generated_image:
                    st.image(generated_image, caption='Generated Image',
                             use_column_width=True)
                    filename = f"{st.session_state.username}_generated_image.png"
                    generated_image.save(filename)
                    st.success(f"Image saved as {filename}")
                    img_data = BytesIO()
                    generated_image.save(img_data, format='PNG')
                    img_data.seek(0)
                    insert_image(st.session_state.user_id,
                                 prompt, model, img_data.read())
            else:
                st.warning("Please enter a prompt.")
    else:
        page = st.sidebar.radio("Navigation", ["Login", "Create Account"])
        if page == "Login":
            login()
        else:
            create_account_page()


if __name__ == '__main__':
    main()
