import streamlit as st
import io
import os
import re
import csv
import zipfile
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
import pillow_avif   # registers AVIF plugin
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# ==================================================
# Init
# ==================================================
register_heif_opener()

# ==============================
#  Load API Keys
# ==============================
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API Key is missing or incorrect! Please update it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==============================
#  Load BLIP Model
# ==============================
@st.cache_resource
def load_blip_model():
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("🔑 Hugging Face token is missing! Please add HUGGINGFACE_TOKEN in Streamlit Secrets.")
        st.stop()

    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name, use_auth_token=hf_token)
    model = BlipForConditionalGeneration.from_pretrained(model_name, use_auth_token=hf_token)

    return processor, model

processor, blip_model = load_blip_model()

# ==================================================
# Helper Functions
# ==================================================
def sanitize_text(text: str) -> str:
    """Remove potentially dangerous characters to avoid prompt injection and filename issues."""
    return re.sub(r"[^\w\s.,'’\-]", "", text)

def blip_generate_caption(image: Image.Image) -> str:
    """Generate a descriptive caption using BLIP with improved settings."""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=3,
            length_penalty=1.0
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()

def has_all_required_elements(text: str, kw_list: list[str], loc: str) -> bool:
    txt_lower = text.lower()
    for kw in kw_list:
        if kw.lower() not in txt_lower:
            return False
    if loc.lower() not in txt_lower:
        return False
    return True

def optimize_alt_tag(
    caption: str,
    keywords: list[str],
    theme: str,
    location: str,
    img_name: str,
    gpt_temperature: float
) -> str:
    """
    Generate optimized alt text under 100 chars including all keywords and location.
    """
    # Sanitize inputs
    caption = sanitize_text(caption)
    theme = sanitize_text(theme)
    location = sanitize_text(location)
    img_name = sanitize_text(img_name)
    keywords = [sanitize_text(k) for k in keywords]

    system_prompt = (
        "You are a strict SEO assistant. You must generate alt text that:\n"
        "• Stays under 100 characters.\n"
        "• Clearly references the subject.\n"
        "• NATURALLY includes all provided keywords and the location.\n"
        "• Omits 'image of' or 'picture of'.\n"
        "• Is useful for visually impaired users.\n"
        "• Return only the final alt text."
    )

    user_prompt = (
        f"BLIP caption: '{caption}'\n"
        f"Filename: '{img_name}'\n"
        f"Keywords: {', '.join(keywords)}\n"
        f"Theme: {theme}\n"
        f"Location: {location}\n\n"
        "Requirements:\n"
        "1. Include all keywords and the location.\n"
        "2. Stay under 100 characters.\n"
        "3. Avoid 'image of' or similar phrases.\n"
        "4. Return ONLY the final alt text."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",     # switch to "gpt-5" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=gpt_temperature,
            max_tokens=80
        )
        alt_tag = response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return caption[:80] + "..."

    # Retry loop
    attempts = 0
    while attempts < 3:
        if len(alt_tag) <= 100 and has_all_required_elements(alt_tag, keywords, location):
            break

        attempts += 1
        retry_prompt = (
            f"Previous alt text: '{alt_tag}'\n"
            f"Required keywords: {keywords}\n"
            f"Required location: {location}\n"
            "Revise to meet all requirements while staying under 100 characters.\n"
            "Return ONLY the new alt text."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",   # switch to "gpt-5" if desired
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_prompt}
                ],
                temperature=gpt_temperature,
                max_tokens=80
            )
            alt_tag = response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Retry failed: {e}")
            break

    # Final fallback
    if len(alt_tag) > 100 or not has_all_required_elements(alt_tag, keywords, location):
        fallback = (caption + " " + " ".join(keywords) + " " + location)[:100]
        st.warning("⚠️ Used fallback alt text due to compliance issues.")
        return fallback

    return alt_tag

def resize_image(image: Image.Image, max_width: int) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(ratio * float(image.height))
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return image

def export_image(image: Image.Image, alt_tag: str, user_format_choice: str):
    """Save the image with sanitized alt_tag as filename."""
    alt_tag_cleaned = sanitize_text(alt_tag).replace(" ", "_")
    if len(alt_tag_cleaned) > 100:
        st.warning("❌ Generated alt text exceeded 100 characters after cleaning.")
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
    except Exception as e:
        st.error(f"Image save error: {e}")
        return None, None

    return img_bytes, filename

# ==================================================
# Streamlit UI
# ==================================================
st.title("🖼️ SEO Image Alt Tag Generator (BLIP + GPT-4o/GPT-5)")

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

if st.button("Reset App"):
    st.session_state.pop("blip_captions", None)
    st.session_state["upload_key"] += 1

st.markdown("""
**Welcome!**  
1. Upload images or a **.zip folder** of images.  
2. Provide **keywords**, **theme**, and **location**.  
3. BLIP generates captions; GPT optimizes them into alt text.  
4. Download a ZIP of renamed images and a CSV metadata file.
""")

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    resize_option = st.checkbox("Resize images before export?")
    if resize_option:
        max_width_setting = st.slider("Max Width (px):", 100, 2000, 800, 50)
    else:
        max_width_setting = None

    user_format_choice = st.selectbox(
        "Output Format:",
        ["PNG", "JPEG", "WEBP", "AVIF"],
        index=0
    )

    gpt_temperature = st.slider(
        "GPT Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

# Upload mode
upload_mode = st.radio("Select upload method:", ["Upload Images", "Upload a .zip Folder of Images"])
all_input_images = []

if upload_mode == "Upload Images":
    uploaded_files = st.file_uploader(
        "Drag & drop or select images",
        type=["jpg", "jpeg", "png", "webp", "avif"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )
    if uploaded_files:
        all_input_images = [(uf.name, uf.read()) for uf in uploaded_files]

elif upload_mode == "Upload a .zip Folder of Images":
    uploaded_zip = st.file_uploader(
        "Drag & drop or select a .zip file",
        type=["zip"],
        key=f"zip_uploader_{st.session_state.upload_key}"
    )
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir() and file_info.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                    base_name = os.path.basename(file_info.filename)
                    all_input_images.append((base_name, zip_ref.read(file_info.filename)))

# ==================================================
# 🟢 Recommended Patch: Safe Threaded Captioning
# ==================================================
if all_input_images:
    st.success(f"**Total Images Found:** {len(all_input_images)}")

    # Ensure key exists
    if "blip_captions" not in st.session_state:
        st.session_state["blip_captions"] = {}

    # Use a local cache for thread safety
    captions_cache = st.session_state["blip_captions"]

    st.markdown("#### Uploaded Images")
    cols = st.columns(3)
    for i, (img_name, img_bytes) in enumerate(all_input_images):
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        cols[i % 3].image(image, caption=img_name, width=150)

    st.markdown("---")
    st.markdown("### Provide Keywords, Theme & Location")
    keywords_input = st.text_input("🔑 Target keywords (comma-separated):", "")
    theme_input = st.text_input("🎨 Theme of the photos:", "")
    location_input = st.text_input("📍 Location (for local SEO):", "")

    if st.button("🚀 Generate & Download Alt-Optimized Images"):
        if not keywords_input.strip() or not theme_input.strip() or not location_input.strip():
            st.warning("Please provide all required fields.")
            st.stop()

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        theme = theme_input.strip()
        location = location_input.strip()

        def process_caption(name, data):
            if name not in captions_cache:
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                return name, blip_generate_caption(pil_img)
            return name, captions_cache[name]

        # Run BLIP captioning in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda x: process_caption(*x), all_input_images))

        # Update session state with results
        for name, caption in results:
            captions_cache[name] = caption

        # Prepare zip and csv
        zip_buffer = io.BytesIO()
        zipf = zipfile.ZipFile(zip_buffer, "w")
        csv_data = [("Original Filename", "BLIP Caption", "Optimized Alt Text", "Alt Text Length", "Exported Filename")]

        for img_name, img_bytes in all_input_images:
            blip_caption = captions_cache[img_name]

            optimized_alt_tag = optimize_alt_tag(
                caption=blip_caption,
                keywords=keywords,
                theme=theme,
                location=location,
                img_name=img_name,
                gpt_temperature=gpt_temperature
            )

            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if resize_option and max_width_setting:
                pil_img = resize_image(pil_img, max_width_setting)

            img_bytes_out, exported_filename = export_image(pil_img, optimized_alt_tag, user_format_choice)
            if img_bytes_out is None:
                continue

            zipf.writestr(exported_filename, img_bytes_out.getvalue())
            csv_data.append((img_name, blip_caption, optimized_alt_tag, len(optimized_alt_tag), exported_filename))

        zipf.close()
        zip_buffer.seek(0)

        st.download_button(
            label="📥 Download ZIP of Optimized Images",
            data=zip_buffer,
            file_name="optimized_images.zip",
            mime="application/zip"
        )

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        for row in csv_data:
            writer.writerow(row)
        st.download_button(
            label="📄 Download CSV Metadata",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="image_metadata.csv",
            mime="text/csv"
        )

        st.markdown("#### Summary Table")
        df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
        st.dataframe(df, use_container_width=True)
