from ast import literal_eval
from io import BytesIO

import streamlit as st
import torch
from database import DatabaseManager
import requests
from PIL import Image

st.set_page_config(layout="wide")
db_manager = DatabaseManager()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DEFAULT_PROMPT = "(1girl:2),(beautiful eyes:1.2),beautiful girl,(hands in pocket:2),red hat,hoodie,computer"
DEFAULT_NEGATIVE_PROMPT = "(bad hands:5),(fused fingers:5),(bad arms:4),(deformities:1.3),(extra limbs:2),(extra arms:2),(disfigured:2)"


def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        authenticated, user_id, username = db_manager.authenticate(username, password)
        if authenticated:
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.query_params.logged_in = True
            st.query_params.user_id = user_id
            st.query_params.username = username
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
                col[j].markdown(
                    f"**_Model:_** {image[2]}<br>**_Seed:_** {literal_eval(image[3])['seed']}<br>**_Prompt:_** {literal_eval(image[3])['prompt']}",
                    unsafe_allow_html=True
                )
                # col[j].image(
                #     image[-1],
                #     caption=f"Model: {image[2]},\tSeed: {literal_eval(image[3])['seed']},\tPrompt: {literal_eval(image[3])['prompt']}",
                # )

    if st.button("Delete Images") and images_to_delete:
        db_manager.delete_images(images_to_delete)
        st.experimental_rerun()


def query_inference_endpoint(**kwargs):
    url = "http://localhost:5000/inference"
    payload = {
        "prompt": kwargs.get("prompt"),
        "negative_prompt": kwargs.get("negative_prompt"),
        "batch_size": kwargs.get("batch_size"),
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "seed": kwargs.get("seed"),
        "mode": kwargs.get("mode"),
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_kwargs["width"] = st.number_input(label="Width", value=1024)
        with col2:
            model_kwargs["height"] = st.number_input(label="Height", value=1024)
        with col3:
            model_kwargs["batch_size"] = st.number_input(
                label="Batch Size", value=3, min_value=1, max_value=9
            )
        with col4:
            model_kwargs["seed"] = st.number_input(label="Seed", value=69420)

    if st.button("Generate images"):
        with st.spinner("Generating image..."):
            output_images = query_inference_endpoint(**model_kwargs)
        if not output_images:
            return
        st.success("Done!")

        num_images = len(output_images)
        num_cols = min(num_images, 3)
        cols = st.columns(num_cols)

        # TODO: improve images layout based on width and height settings
        max_width = int(model_kwargs["width"])
        for i, image in enumerate(output_images):
            with cols[i % num_cols]:
                st.image(
                    image,
                    width=max_width,
                    caption=f"Image {i+1}",
                    use_column_width=True,
                )
                img_data = BytesIO()
                image.save(img_data, format="PNG")
                img_data.seek(0)
                db_manager.insert_image(
                    st.session_state.user_id,
                    f"{model_kwargs['mode']}",
                    model_kwargs,
                    img_data.read(),
                )


def main():
    query_params = st.query_params
    logged_in = query_params.get("logged_in", False)
    user_id = query_params.get("user_id", None)
    username = query_params.get("username", None)

    st.session_state.logged_in = False
    if logged_in and user_id and username:
        st.session_state.logged_in = True
        st.session_state.user_id = int(user_id)
        st.session_state.username = username

    if st.session_state.logged_in:
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
