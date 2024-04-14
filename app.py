from ast import literal_eval
from io import BytesIO

import streamlit as st
import io
import torch
from database import DatabaseManager
import requests
from PIL import Image

st.set_page_config(layout="wide")
db_manager = DatabaseManager()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DEFAULT_PROMPT = "(1girl:2),(beautiful eyes:1.2),beautiful girl,red hat,hoodie"
DEFAULT_NEGATIVE_PROMPT = "(bad hands:5),(fused fingers:5),(bad arms:4),(deformities:1.3),(extra limbs:2),(extra arms:2),(disfigured:2)"


def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        authenticated, user_id, username = db_manager.authenticate(username, password)
        if authenticated:
            st.session_state.username = username
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")
    return False


def create_account_page():
    st.title("Create Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        if new_username.strip() and new_password.strip():
            error = db_manager.create_account(new_username, new_password)
            if error:
                st.error(error)
                return
            st.success("Account created successfully!")
            st.write("You can now login with your new account.")
        else:
            st.error("Username and password cannot be empty.")


def profile_page():
    all_users = db_manager.get_users()
    user_names = [u[1] for u in all_users]
    user_id_map = {u[1]: u[0] for u in all_users}

    current_user_index = user_names.index(st.session_state.username)

    selected_user = st.selectbox("Select Profile", user_names, index=current_user_index)
    selected_user_id = user_id_map[selected_user]

    user_images = db_manager.get_user_images(selected_user_id)
    is_owner = st.session_state.user_id == selected_user_id

    st.header(f"{'Your' if is_owner else selected_user} Profile")

    if len(user_images) == 0:
        st.write("This user doesn't have any generated images.")
        return

    num_columns = 4
    images_to_delete = []
    images_per_row = len(user_images) // num_columns + 1
    for i in range(images_per_row):
        col = st.columns(num_columns)
        for j in range(num_columns):
            index = i * num_columns + j
            if index < len(user_images):
                image = user_images[index]
                if is_owner:
                    delete_checkbox = col[j].checkbox(
                        f"Delete Image {image[0]}", key=f"delete_{image[0]}"
                    )
                    if delete_checkbox:
                        images_to_delete.append(image[0])
                col[j].image(
                    image[-1],
                )
                step_str = (
                    f"**_Steps:_** Base ({literal_eval(image[3])['base_step']}) | Refiner ({literal_eval(image[3])['refiner_step']})<br>"
                    if "base_step" in literal_eval(image[3]).keys()
                    else ""
                )
                cfg_str = (
                    f"**_Guidance scale:_** {literal_eval(image[3])['cfg']}<br>"
                    if "cfg" in literal_eval(image[3]).keys()
                    else ""
                )
                img2img_str = (
                    f"**_Image to Image mode_**<br>"
                    if step_str == "" and cfg_str == ""
                    else ""
                )
                col[j].markdown(
                    f"**_Model:_** {image[2]}<br>\
                    {step_str}\
                    {cfg_str}\
                    {img2img_str}\
                    **_Seed:_** {literal_eval(image[3])['seed']}<br>\
                    **_Prompt:_** {literal_eval(image[3])['prompt']}\
                    ",
                    unsafe_allow_html=True,
                )
                # col[j].image(
                #     image[-1],
                #     caption=f"Model: {image[2]},\tSeed: {literal_eval(image[3])['seed']},\tPrompt: {literal_eval(image[3])['prompt']}",
                # )

    if st.button("Delete Images") and images_to_delete:
        db_manager.delete_images(images_to_delete)
        st.rerun()


def request_text2image(**kwargs):
    url = "http://localhost:5000/text2img"
    payload = {
        "prompt": kwargs.get("prompt"),
        "negative_prompt": kwargs.get("negative_prompt"),
        "batch_size": kwargs.get("batch_size"),
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "seed": kwargs.get("seed"),
        "mode": kwargs.get("mode"),
        "base_step": kwargs.get("base_step"),
        "refiner_step": kwargs.get("refiner_step"),
        "cfg": kwargs.get("cfg"),
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        image_bytes_stream = response.content
        delimiter = b"--DELIMITER--"
        image_bytes_list = image_bytes_stream.split(delimiter)
        images = []
        for image_bytes in image_bytes_list:
            if image_bytes:
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
        return images
    else:
        st.error(f"Error: {response.status_code}")
        return None


def request_image2image(**kwargs):
    url = "http://localhost:5000/img2img"
    payload = {
        "prompt": kwargs.get("prompt"),
        "negative_prompt": kwargs.get("negative_prompt", ""),
        "seed": kwargs.get("seed"),
        "mode": kwargs.get("mode"),
    }
    headers = {"Content-Type": "application/octet-stream"}
    response = requests.post(
        url, data=kwargs.get("image"), headers=headers, params=payload
    )

    if response.status_code == 200:
        image_bytes_stream = response.content
        delimiter = b"--DELIMITER--"
        image_bytes_list = image_bytes_stream.split(delimiter)
        images = []
        for image_bytes in image_bytes_list:
            if image_bytes:
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
        return images
    else:
        st.error(f"Error: {response.status_code} {response.text}")
        return None


# Specify the path to the default image file
default_image_path = "images/demo.jpg"

# Read the default image file as bytes
with open(default_image_path, "rb") as file:
    default_image_bytes = file.read()


def generate_page():
    st.subheader("Generate and Save Image 👨‍🎨")
    model_kwargs = {
        "mode": st.selectbox("Mode", ["anime", "cartoon", "realistic"]),
        "prompt": st.text_input(
            "Prompt",
            placeholder=DEFAULT_PROMPT,
            value=DEFAULT_PROMPT,
            help="Your desired features for the generated images",
        ),
    }

    with st.expander("Refine your output"):
        model_kwargs["negative_prompt"] = st.text_input(
            "Negative Prompt",
            placeholder=DEFAULT_NEGATIVE_PROMPT,
            help="Features you want to exclude from the image",
        )
        # Create a file uploader widget with the default value
        uploaded_file = st.file_uploader(
            "Use Image to Image (img2img) workflow",
            type=["jpg", "jpeg", "png"],
            help="Refine your source image with additional styles. Recommend to use with minimal prompts.",
        )
        # Read the uploaded file as bytes
        if uploaded_file:
            file_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(file_bytes))

            # Display the uploaded image
            st.image(file_bytes, caption="Source Image", width=min(image.width, 500))

            model_kwargs["image"] = file_bytes

            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input(label="Width", value=image.width, disabled=True)
            with col2:
                st.number_input(label="Height", value=image.height, disabled=True)
            with col3:
                model_kwargs["seed"] = st.number_input(label="Seed", value=4052)

        else:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                model_kwargs["width"] = st.number_input(label="Width", value=1024)
            with col2:
                model_kwargs["height"] = st.number_input(label="Height", value=1024)
            with col3:
                model_kwargs["batch_size"] = st.number_input(
                    label="Batch Size", value=3, min_value=1, max_value=9
                )
            with col4:
                model_kwargs["seed"] = st.number_input(label="Seed", value=4052)
            with col5:
                model_kwargs["base_step"] = st.number_input(
                    label="Base Step", value=8, help="Steps on base model"
                )
            with col6:
                model_kwargs["refiner_step"] = st.number_input(
                    label="Refiner Step", value=1, help="Steps on refiner (if any)"
                )
            with col7:
                model_kwargs["cfg"] = st.number_input(
                    label="Guidance Scale",
                    value=1.0,
                    help="How strict model should follows prompt (DEFAULT IS NORMALLY THE BEST)",
                )

    if st.button("Generate images"):
        with st.spinner("Generating image..."):
            if uploaded_file:
                output_images = request_image2image(**model_kwargs)
                pass
            else:
                output_images = request_text2image(**model_kwargs)

        if not output_images:
            return
        st.success("Done!")

        num_images = len(output_images)
        num_cols = max(2, min(num_images, 4))
        cols = st.columns(num_cols)

        for i, image in enumerate(output_images):
            with cols[i % num_cols]:
                st.image(
                    image,
                    caption=f"Image {i+1}",
                )
                img_data = BytesIO()
                image.save(img_data, format="PNG")
                img_data.seek(0)
                if not uploaded_file:
                    db_manager.insert_image(
                        st.session_state.user_id,
                        f"{model_kwargs['mode']}",
                        {k: v for k, v in model_kwargs.items() if k != "image"},
                        img_data.read(),
                    )


def main():
    logged_in = st.session_state.get("logged_in", False)
    user_id = st.session_state.get("user_id", None)
    username = st.session_state.get("username", None)

    if logged_in and user_id and username:
        st.session_state.logged_in = True
        st.session_state.user_id = int(user_id)
        st.session_state.username = username
        st.sidebar.write(f"Welcome, {st.session_state.username}! ✨")
        page = st.sidebar.radio(
            "What would you like to do today?", ["Generate Image", "View Profile"]
        )
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
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


if __name__ == "__main__":
    main()
