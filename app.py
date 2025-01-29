import streamlit as st
import openai
from PIL import Image
import io
import base64
import zipfile
import os
import tkinter as tk
from tkinter import filedialog

# Load OpenAI API key securely from Streamlit secrets
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API Key is missing or incorrect! Please update it in Streamlit Secrets.")
    st.stop()

# Initialize OpenAI client with correct API key
openai_client = openai.OpenAI(api_key=api_key)

# Function to encode image as base64
def encode_image_to_base64(image):
    """Convert image to base64 string."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return img_base64

# Function to get image caption using GPT-4 Turbo Vision
def generate_caption_with_gpt4(image):
    """Send image to GPT-4 Turbo Vision and get a description."""
    img_base64 = encode_image_to_base64(image)

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI image captioning assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}
        ],
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

# Function to optimize alt text using GPT-4 Turbo (with SEO best practices)
def optimize_alt_tag_gpt4(caption, keywords, theme):
    """Generate an SEO-optimized alt tag using GPT-4 Turbo with best practices."""
    prompt = (
        f"Here is an image caption: '{caption}'.\n"
        f"The target keywords are: {', '.join(keywords)}.\n"
        f"The theme of the photos is: {theme}.\n"
        f"Please generate an optimized alt text following these SEO best practices:\n"
        f"1️⃣ **Keep it concise**: Alt text should be under 100 characters.\n"
        f"2️⃣ **Provide context**: Describe what the image means in the site's content.\n"
        f"3️⃣ **Include relevant keywords** naturally, without stuffing.\n"
        f"4️⃣ **Avoid unnecessary phrases** like 'image of' or 'picture of'.\n"
        f"5️⃣ **Improve accessibility**: Ensure clarity for visually impaired users.\n"
        f"Generate an optimized alt tag based on these guidelines."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Function to open a folder selection dialog
def select_folder():
    """Opens a system dialog for selecting a folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Destination Folder")
    return folder_path

# Streamlit UI
st.title("🖼️ SEO Image Alt Tag Generator (Supports Single & Multiple Images)")
st.write("Upload images to generate AI-powered alt tags optimized for SEO!")
st.write("Modern Practice Internal Tool")


# User selects if they want to upload a single or multiple images
upload_mode = st.radio("Choose Upload Mode:", ["Single Image", "Multiple Images"])

# Allow single or multiple image uploads based on user choice
uploaded_files = st.file_uploader("📤 Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=(upload_mode == "Multiple Images"))

# Folder selection button
if st.button("📂 Select Destination Folder"):
    selected_folder = select_folder()
    if selected_folder:
        st.session_state["destination_folder"] = selected_folder

# Display selected folder
if "destination_folder" in st.session_state:
    st.success(f"📁 Selected Destination Folder: {st.session_state['destination_folder']}")

if uploaded_files:
    if upload_mode == "Single Image":
        uploaded_files = [uploaded_files]  # Convert to list for consistency

    # Store captions to prevent unnecessary re-processing
    if "image_captions" not in st.session_state:
        st.session_state.image_captions = {}

    # Layout images in a row for multiple image uploads
    col1, col2, col3 = st.columns(3)
    
    # Store optimized images for bulk download
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert('RGB')

            # Generate and store caption only if not already generated
            if uploaded_file.name not in st.session_state.image_captions:
                with st.spinner(f"🔍 Generating caption for {uploaded_file.name}..."):
                    st.session_state.image_captions[uploaded_file.name] = generate_caption_with_gpt4(image)

            # Display images in a row
            if idx % 3 == 0:
                col = col1
            elif idx % 3 == 1:
                col = col2
            else:
                col = col3

            col.image(image, caption=uploaded_file.name, use_container_width=False, width=150)

    keywords = st.text_input("🔑 Enter target keywords (comma-separated)").split(",")
    theme = st.text_input("🎨 Enter the theme of the photos")

    if st.button("🚀 Generate Optimized Alt Tags"):
        if not keywords or not theme:
            st.warning("⚠️ Please enter both keywords and theme.")
        else:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert('RGB')
                basic_caption = st.session_state.image_captions[uploaded_file.name]

                with st.spinner(f"✨ Optimizing Alt Tag for {uploaded_file.name}..."):
                    optimized_alt_tag = optimize_alt_tag_gpt4(basic_caption, keywords, theme)

                st.success(f"✅ Optimized Alt Tag for **{uploaded_file.name}**:")
                st.write(optimized_alt_tag)

                # Export image with new filename
                img_bytes, filename = export_image(image, optimized_alt_tag)
                
                # Save to ZIP for bulk download
                zipf.writestr(filename, img_bytes.getvalue())

    # Provide bulk download for multiple images
    zip_buffer.seek(0)
    st.download_button(
        label="📥 Download All Images",
        data=zip_buffer,
        file_name="optimized_images.zip",
        mime="application/zip"
    )
