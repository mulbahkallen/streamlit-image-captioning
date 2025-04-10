import streamlit as st
import openai
from PIL import Image
import pillow_heif  # Replaces pillow_avif
pillow_heif.register_avif_opener()  # Enable AVIF read/write support

import io
import base64
import zipfile
import os
import csv
import pandas as pd
import re

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==============================
#  Load OpenAI API Key
# ==============================
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API Key is missing or incorrect! Please update it in Streamlit Secrets.")
    st.stop()

# Initialize OpenAI
openai.api_key = api_key

# ==============================
#  Load BLIP Model (cached) -- Updated with Hugging Face Auth Token
# ==============================
@st.cache_resource
def load_blip_model():
    # Pull the HF token from your Streamlit secrets
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("🔑 Hugging Face token is missing! Please add HUGGINGFACE_TOKEN in Streamlit Secrets.")
        st.stop()

    model_name = "Salesforce/blip-image-captioning-base"
    
    # Pass use_auth_token=hf_token to both from_pretrained calls
    processor = BlipProcessor.from_pretrained(model_name, use_auth_token=hf_token)
    model = BlipForConditionalGeneration.from_pretrained(model_name, use_auth_token=hf_token)

    return processor, model

processor, blip_model = load_blip_model()

def blip_generate_caption(image: Image.Image) -> str:
    """
    Uses BLIP to generate an actual descriptive caption for the image.
    """
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()

# ==============================
#  Helper / Utility Functions
# ==============================

def optimize_alt_tag_gpt4(
    caption: str,
    keywords: list[str],
    theme: str,
    location: str,
    img_name: str,
    gpt_temperature: float
) -> str:
    """
    Improved version of the alt text generator:
      - Must stay under 100 characters.
      - Must include all keywords & location.
      - Avoid phrases like 'image of'.
      - Provide clarity for visually impaired.
      - Re-run if missing any requirements.
      - Uses user-selected GPT temperature for "creativity".
    """

    # 1) Stricter system prompt
    system_prompt = (
        "You are a strict SEO assistant. You must generate alt text that:\n"
        " • Stays under 100 characters.\n"
        " • Clearly references the subject.\n"
        " • NATURALLY includes *all* provided keywords and the location.\n"
        " • Omits 'image of' or 'picture of'.\n"
        " • Is helpful to visually impaired users.\n"
        "The alt text must not exceed 100 characters under any circumstances."
    )

    # 2) User prompt emphasizing mandatory elements
    user_prompt = (
        f"Please create a single, concise alt text.\n\n"
        f"Context:\n"
        f"- BLIP caption of the image: '{caption}'\n"
        f"- Filename: '{img_name}'\n"
        f"- Target keywords: {', '.join(keywords)}\n"
        f"- Theme: {theme}\n"
        f"- Location: {location}\n\n"
        f"Mandatory Requirements:\n"
        f"1. Must include every keyword from the target keywords.\n"
        f"2. Must include the location.\n"
        f"3. Must remain under 100 characters total.\n"
        f"4. Avoid phrases like 'image of' or 'picture of'.\n"
        f"5. Return ONLY the final alt text (no commentary)."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=80,
            temperature=gpt_temperature  # <-- Use user-selected creativity
        )
        alt_tag = response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return caption[:80] + "..."

    # Helper to check if all required keywords & location are included
    def has_all_required_elements(text: str, kw_list: list[str], loc: str) -> bool:
        txt_lower = text.lower()
        for kw in kw_list:
            if kw.lower() not in txt_lower:
                return False
        if loc.lower() not in txt_lower:
            return False
        return True

    # Re-run if text is too long or missing mandatory elements
    attempts = 0
    while attempts < 3:
        if len(alt_tag) <= 100 and has_all_required_elements(alt_tag, keywords, location):
            break
        attempts += 1

        shortened_system_prompt = (
            "You must revise the alt text. It is either over 100 characters "
            "or missing required items. All keywords and the location MUST appear, "
            "while staying under 100 characters total."
        )
        shortened_user_prompt = (
            f"Original attempt: '{alt_tag}'\n\n"
            f"Required keywords: {keywords}\n"
            f"Required location: {location}\n\n"
            f"Return ONLY the new alt text under 100 chars."
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": shortened_system_prompt},
                    {"role": "user", "content": shortened_user_prompt}
                ],
                max_tokens=60,
                temperature=gpt_temperature  # Keep same creativity in re-run
            )
            alt_tag = response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            st.error(f"OpenAI API error: {e}")
            break

    return alt_tag

def resize_image(image: Image.Image, max_width: int) -> Image.Image:
    """If the image is wider than max_width, resize it (preserving aspect ratio)."""
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(ratio * float(image.height))
        image = image.resize((max_width, new_height), Image.ANTIALIAS)
    return image

def export_image(
    image: Image.Image,
    alt_tag: str,
    user_format_choice: str
):
    """
    Save the image using the alt_tag for the filename (sanitized).
    Respect the chosen output format (PNG, JPEG, WEBP, AVIF).
    Return (img_bytes, filename) or (None, None) on failure.
    """
    # Clean alt_tag for filename
    alt_tag_cleaned = (
        alt_tag.strip()
              .replace('"', "")
              .replace("'", "")
              .replace(" ", "_")
              .replace(",", "")
              .replace(".", "")
              .replace("/", "")
              .replace("\\", "")
    )
    # Remove leading non-alphanumeric
    alt_tag_cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', alt_tag_cleaned)

    # Ensure <= 100 chars
    if len(alt_tag_cleaned) > 100:
        st.warning(
            f"❌ Generated alt text exceeded 100 characters after cleaning:\n"
            f"'{alt_tag_cleaned}'\nPlease regenerate or manually shorten."
        )
        return None, None

    format_mapping = {
        "PNG":  ("png",  "PNG"),
        "JPEG": ("jpg",  "JPEG"),
        "WEBP": ("webp", "WEBP"),
        "AVIF": ("avif", "AVIF")
    }
    extension, pillow_format = format_mapping[user_format_choice]
    filename = f"{alt_tag_cleaned}.{extension}"

    img_bytes = io.BytesIO()
    try:
        image.save(img_bytes, format=pillow_format)
        img_bytes.seek(0)
    except ValueError as e:
        st.error(f"Image save error: {e}")
        return None, None

    return img_bytes, filename

# ==============================
#  Streamlit UI
# ==============================
st.title("🖼️ SEO Image Alt Tag Generator (BLIP + GPT-4)")

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

# Reset App Button
if st.button("Reset App"):
    if "blip_captions" in st.session_state:
        del st.session_state["blip_captions"]
    st.session_state["upload_key"] += 1

st.markdown("""
**Welcome!**  
1. Upload or drag-and-drop images (individual or multiple).  
2. Alternatively, upload a **.zip folder** of images.  
3. Provide your **keywords**, **theme**, and **location** for local SEO.  
4. A BLIP model will first describe your image. Then GPT-4 will generate a final alt tag.  
5. Download a ZIP with renamed images **and** a CSV file for metadata.
""")

# --- Advanced Settings ---
with st.expander("⚙️ Advanced Settings"):
    st.markdown("**Resize Images** (to unify widths) and choose **Output Format**.")
    resize_option = st.checkbox("Resize images before export?")
    if resize_option:
        max_width_setting = st.slider("Max Width (px):", 100, 2000, 800, 50)
    else:
        max_width_setting = None

    user_format_choice = st.selectbox(
        "Output Format for Exported Images:",
        ["PNG", "JPEG", "WEBP", "AVIF"],
        index=0
    )

    # New: GPT Creativity
    gpt_temperature = st.slider(
        "GPT-4 Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # default
        step=0.1
    )

# Let user choose how to provide images
upload_mode = st.radio(
    "How would you like to provide images?",
    ["Upload Images", "Upload a .zip Folder of Images"]
)

all_input_images = []

# ========== Option 1: Upload Images ==========
if upload_mode == "Upload Images":
    uploaded_files = st.file_uploader(
        "Drag & drop or select images",
        type=["jpg", "jpeg", "png", "webp", "avif"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )
    if uploaded_files:
        for uf in uploaded_files:
            all_input_images.append((uf.name, uf.read()))

# ========== Option 2: Upload a .zip ==========
elif upload_mode == "Upload a .zip Folder of Images":
    uploaded_zip = st.file_uploader(
        "Drag & drop or select a .zip file of images",
        type=["zip"],
        accept_multiple_files=False,
        key=f"zip_uploader_{st.session_state.upload_key}"
    )
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    filename_lower = file_info.filename.lower()
                    if filename_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                        file_bytes = zip_ref.read(file_info.filename)
                        base_name = os.path.basename(file_info.filename)
                        all_input_images.append((base_name, file_bytes))

# If images are provided, show them
if all_input_images:
    st.success(f"**Total Images Found**: {len(all_input_images)}")

    if "blip_captions" not in st.session_state:
        st.session_state["blip_captions"] = {}

    # Display the uploaded images as thumbnails
    st.markdown("#### Uploaded Images")
    colA, colB, colC = st.columns(3)
    for i, (img_name, img_bytes_data) in enumerate(all_input_images):
        image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
        if i % 3 == 0:
            col = colA
        elif i % 3 == 1:
            col = colB
        else:
            col = colC
        col.image(image, caption=img_name, width=150)

    st.markdown("---")
    st.markdown("### Provide Keywords, Theme, & Location")
    st.markdown("These will help GPT-4 optimize the alt text for SEO, including local context.")

    keywords_input = st.text_input("🔑 Target keywords (comma-separated):", "")
    theme_input = st.text_input("🎨 Theme of the photos:", "")
    location_input = st.text_input("📍 Location (for local SEO):", "")

    if st.button("🚀 Generate & Download Alt-Optimized Images"):
        if not keywords_input.strip() or not theme_input.strip() or not location_input.strip():
            st.warning("Please provide keywords, theme, and location to proceed.")
            st.stop()

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        theme = theme_input.strip()
        location = location_input.strip()

        # Create in-memory ZIP
        zip_buffer = io.BytesIO()
        zipf = zipfile.ZipFile(zip_buffer, "w")

        # Prepare CSV data
        csv_data = [
            ("Original Filename", "BLIP Caption", "Optimized Alt Text", "Alt Text Length", "Exported Filename")
        ]

        for img_name, img_bytes_data in all_input_images:
            # Step A: BLIP-based caption
            if img_name not in st.session_state["blip_captions"]:
                with st.spinner(f"Generating BLIP caption for {img_name}..."):
                    pil_img = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
                    blip_caption = blip_generate_caption(pil_img)
                    st.session_state["blip_captions"][img_name] = blip_caption

            blip_caption = st.session_state["blip_captions"][img_name]

            # Step B: GPT-4 alt text
            with st.spinner(f"Optimizing alt text for {img_name}..."):
                optimized_alt_tag = optimize_alt_tag_gpt4(
                    caption=blip_caption,
                    keywords=keywords,
                    theme=theme,
                    location=location,
                    img_name=img_name,
                    gpt_temperature=gpt_temperature  # <-- Pass user-selected temperature
                )

            # Step C: Resize if needed
            pil_img = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
            if resize_option and max_width_setting:
                pil_img = resize_image(pil_img, max_width_setting)

            # Step D: Export the image
            img_bytes, exported_filename = export_image(pil_img, optimized_alt_tag, user_format_choice)
            if img_bytes is None or exported_filename is None:
                continue

            # Write to ZIP
            zipf.writestr(exported_filename, img_bytes.getvalue())

            # Collect CSV row
            alt_len = len(optimized_alt_tag)
            csv_data.append((img_name, blip_caption, optimized_alt_tag, str(alt_len), exported_filename))

        zipf.close()
        zip_buffer.seek(0)

        # ---- Download ZIP
        st.download_button(
            label="📥 Download ZIP of Optimized Images",
            data=zip_buffer,
            file_name="optimized_images.zip",
            mime="application/zip"
        )

        # Create CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        for row in csv_data:
            writer.writerow(row)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        # ---- Download CSV
        st.download_button(
            label="📄 Download CSV Metadata",
            data=csv_bytes,
            file_name="image_metadata.csv",
            mime="text/csv"
        )

        # ---- Display Summary Table
        st.markdown("#### Summary Table")
        headers = csv_data[0]
        rows = csv_data[1:]
        df = pd.DataFrame(rows, columns=headers)
        st.dataframe(df, use_container_width=True)
