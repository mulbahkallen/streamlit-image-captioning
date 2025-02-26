import streamlit as st
import openai
from PIL import Image
import pillow_avif  # Registers AVIF support
import io
import base64
import zipfile
import os
import csv
import pandas as pd
import re  # For regex

# ==============================
#  Load OpenAI API Key
# ==============================
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API Key is missing or incorrect! Please update it in Streamlit Secrets.")
    st.stop()

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=api_key)

# ==============================
#  Helper / Utility Functions
# ==============================
@st.cache_data
def encode_image_to_base64(image):
    """
    Convert PIL image to base64 string.
    Caching helps avoid re-encoding the same image on repeated runs.
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")  # internally for GPT vision
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return img_base64

@st.cache_data
def generate_caption_with_gpt4(image_bytes):
    """
    Send image bytes to GPT-4 Turbo Vision and get a description.
    This is cached so repeated runs on the same image won't cost new tokens.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_base64 = encode_image_to_base64(image)

    # Make GPT-4 request
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI image captioning assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

@st.cache_data
def optimize_alt_tag_gpt4(caption, keywords, theme, location):
    """
    Generate an SEO-optimized alt tag using GPT-4 Turbo with best practices.
    Cached to avoid repeated calls with identical inputs.
    """
    prompt = (
        f"Image caption: '{caption}'.\n"
        f"Target keywords: {', '.join(keywords)}.\n"
        f"Theme: {theme}.\n"
        f"Location: {location}.\n\n"
        f"Please generate an optimized alt text following these SEO best practices:\n"
        f"1️⃣ Keep it concise (under 100 characters).\n"
        f"2️⃣ Provide context relevant to the site's content.\n"
        f"3️⃣ Naturally include the keywords and location without stuffing.\n"
        f"4️⃣ Avoid 'image of', 'picture of', or similar phrases.\n"
        f"5️⃣ Provide clarity for visually impaired users.\n\n"
        f"Generate a final optimized alt tag now."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

def resize_image(image, max_width):
    """
    Optionally resize the image if its width is larger than max_width,
    preserving aspect ratio.
    """
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(ratio * float(image.height))
        image = image.resize((max_width, new_height), Image.ANTIALIAS)
    return image

def export_image(image, alt_tag, user_format_choice):
    """
    Save the image with the optimized alt tag as the filename.
    Also respect the chosen output format (e.g. PNG, JPEG, WEBP, AVIF).
    """
    alt_tag_cleaned = (
        alt_tag.replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
               .replace("/", "")
               .replace("\\", "")
    )

    # ---------- Strip leading special characters ----------
    alt_tag_cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', alt_tag_cleaned)

    # ---------- Enforce 100-character limit ---------------
    if len(alt_tag_cleaned) > 100:
        st.warning(
            f"❌ Cannot save image because filename (derived from alt tag) exceeds "
            f"100 characters:\n'{alt_tag_cleaned}'"
        )
        return None, None

    # Map from user choice (PNG/JPEG/WEBP/AVIF) to file extension and Pillow format
    format_mapping = {
        "PNG":  ("png",  "PNG"),
        "JPEG": ("jpg",  "JPEG"),
        "WEBP": ("webp", "WEBP"),
        "AVIF": ("avif", "AVIF")
    }

    # Extract the extension and Pillow format
    extension, pillow_format = format_mapping[user_format_choice]
    filename = f"{alt_tag_cleaned}.{extension}"

    # Save the image to a BytesIO buffer
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=pillow_format)
    img_bytes.seek(0)

    return img_bytes, filename

# ==============================
#  Streamlit UI
# ==============================
st.title("🖼️ SEO Image Alt Tag Generator")

# 1. Initialize or update upload_key in session state if needed
if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

# 2. Reset App Button
if st.button("Reset App"):
    # Clear GPT-4 captions if you want to (optional)
    if "image_captions" in st.session_state:
        del st.session_state["image_captions"]
    # Increment the key so we get a fresh uploader
    st.session_state["upload_key"] += 1
    # Note: We do NOT call st.experimental_rerun()
    # Because a button press automatically triggers a rerun anyway.

st.markdown("""
**Welcome!**  
1. Drag-and-drop individual images or multiple images.  
2. Alternatively, upload a **.zip folder** of images if you have a folder of images.  
3. Provide y
