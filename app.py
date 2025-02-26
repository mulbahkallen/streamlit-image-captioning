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
import re

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
#  Helper / Utility Functions
# ==============================

def generate_caption_with_gpt4(img_name: str) -> str:
   
    system_prompt = "You are a helpful assistant that creates short image captions."
    user_prompt = (
        f"Please create a short, generic description for an image. "
        f"The filename is '{img_name}'. "
        f"Try to infer minimal context or theme from the filename. "
        f"Return a concise, natural-sounding caption."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return "Default placeholder caption"

def re_run_shortening_gpt4(
    too_long_text: str,
    caption: str,
    keywords: list[str],
    theme: str,
    location: str,
    img_name: str
) -> str:
    """
    If the alt text is too long (>100 chars), run GPT-4 again with a stricter prompt
    to shorten it while retaining key context.
    """
    system_prompt = "You are a helpful assistant that shortens alt text under strict constraints."
    user_prompt = (
        f"The following alt text is {len(too_long_text)} characters, but it must be under 100.\n"
        f"Original alt text: '{too_long_text}'\n\n"
        f"Please revise it to be strictly under 100 characters, while retaining the SEO essence.\n\n"
        f"(Context)\n"
        f"Caption: '{caption}'\n"
        f"Target keywords: {', '.join(keywords)}\n"
        f"Theme: {theme}\n"
        f"Location: {location}\n"
        f"Filename: {img_name}\n\n"
        f"Return ONLY the shortened alt text under 100 characters."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        # Fallback: just return the original
        return too_long_text

def optimize_alt_tag_gpt4(
    caption: str,
    keywords: list[str],
    theme: str,
    location: str,
    img_name: str
) -> str:
    """
    Generate an SEO-optimized alt text using GPT-4 with best practices:
      1) Under 100 characters
      2) Provide relevant context
      3) Include keywords & location
      4) Avoid 'image of', 'picture of'
      5) Clarity for visually impaired
      6) Use filename context for uniqueness
    """
    system_prompt = "You are a helpful SEO assistant creating concise alt text."
    user_prompt = (
        f"Image caption: '{caption}'.\n"
        f"Filename: '{img_name}'.\n"
        f"Target keywords: {', '.join(keywords)}.\n"
        f"Theme: {theme}.\n"
        f"Location: {location}.\n\n"
        f"Please generate an optimized alt text following these SEO best practices:\n"
        f"1️⃣ Under 100 characters.\n"
        f"2️⃣ Provide context relevant to the site's content.\n"
        f"3️⃣ Naturally include the keywords and location.\n"
        f"4️⃣ Avoid phrases like 'image of', 'picture of'.\n"
        f"5️⃣ Provide clarity for visually impaired users.\n\n"
        f"Return ONLY the final alt text."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        # fallback
        return caption[:80] + "..."

    alt_tag = response.choices[0].message.content.strip()

    # Re-run up to 3 times if GPT doesn't respect the length
    for _ in range(3):
        if len(alt_tag) <= 100:
            break
        alt_tag = re_run_shortening_gpt4(
            too_long_text=alt_tag,
            caption=caption,
            keywords=keywords,
            theme=theme,
            location=location,
            img_name=img_name
        )

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
) -> tuple[io.BytesIO, str] | tuple[None, None]:
    """
    Save the image using the alt_tag for the filename (sanitized).
    Respect the chosen output format (PNG, JPEG, WEBP, AVIF).
    Return (img_bytes, filename) or (None, None) on failure.
    """
    # Clean alt_tag for filename
    alt_tag = alt_tag.strip('"').strip("'")
    alt_tag_cleaned = (
        alt_tag.replace('"', "")
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

    # Map user choice to Pillow format
    format_mapping = {
        "PNG":  ("png",  "PNG"),
        "JPEG": ("jpg",  "JPEG"),
        "WEBP": ("webp", "WEBP"),
        "AVIF": ("avif", "AVIF")
    }
    extension, pillow_format = format_mapping[user_format_choice]
    filename = f"{alt_tag_cleaned}.{extension}"

    # Save image to in-memory buffer
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=pillow_format)
    img_bytes.seek(0)

    return img_bytes, filename

# ==============================
#  Streamlit UI
# ==============================
st.title("🖼️ SEO Image Alt Tag Generator")

# Session key for reloading the uploader
if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

# Reset App Button
if st.button("Reset App"):
    if "image_captions" in st.session_state:
        del st.session_state["image_captions"]
    st.session_state["upload_key"] += 1

st.markdown("""
**Welcome!**  
1. Upload or drag-and-drop images (individual or multiple).  
2. Alternatively, upload a **.zip folder** of images.  
3. Provide your **keywords**, **theme**, and **location** for local SEO.   
4. Download a ZIP with renamed images **and** a CSV metadata file.
""")

# --- Advanced Settings ---
with st.expander("⚙️ Advanced Settings"):
    st.markdown("**Resize Images** (to unify widths) and choose **Output Format**.")
    resize_option = st.checkbox("Resize images before export?")
    if resize_option:
        max_width_setting = st.slider("Max Width (px):", 100, 2000, 800, 50)
    else:
        max_width_setting = None

    # Let user pick output format
    user_format_choice = st.selectbox(
        "Output Format for Exported Images:",
        ["PNG", "JPEG", "WEBP", "AVIF"],
        index=0
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

    # If we haven't stored captions in session yet, do so now
    if "image_captions" not in st.session_state:
        st.session_state["image_captions"] = {}

    # ============ Step 1: Show Thumbnails (before generation) ============
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

    # ============ Step 2: Ask for SEO Inputs ============
    st.markdown("---")
    st.markdown("### Provide Keywords, Theme, & Location")
    st.markdown("These will help GPT-4 optimize the alt text for SEO, including local context.")

    keywords_input = st.text_input("🔑 Target keywords (comma-separated):", "")
    theme_input = st.text_input("🎨 Theme of the photos:", "")
    location_input = st.text_input("📍 Location (for local SEO):", "")

    # ============ Step 3: Button to Generate & Show Alt Text Preview ============
    if st.button("🚀 Generate & Preview Alt Texts"):
        if not keywords_input.strip() or not theme_input.strip() or not location_input.strip():
            st.warning("Please provide keywords, theme, and location to proceed.")
            st.stop()

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        theme = theme_input.strip()
        location = location_input.strip()

        # We'll store tuples of (img_name, resized_image, alt_text)
        preview_data = []

        st.markdown("---")
        st.markdown("#### GPT-4 Alt Text Preview")
        col1, col2, col3 = st.columns(3)

        for idx, (img_name, img_bytes_data) in enumerate(all_input_images):
            # Generate or retrieve GPT-4 placeholder caption
            if img_name not in st.session_state["image_captions"]:
                with st.spinner(f"Generating placeholder caption for {img_name}..."):
                    st.session_state["image_captions"][img_name] = generate_caption_with_gpt4(img_name)

            basic_caption = st.session_state["image_captions"][img_name]

            # Optimize alt text
            with st.spinner(f"Optimizing alt text for {img_name}..."):
                optimized_alt_tag = optimize_alt_tag_gpt4(
                    caption=basic_caption,
                    keywords=keywords,
                    theme=theme,
                    location=location,
                    img_name=img_name
                )

            # Resize if needed
            image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
            if resize_option and max_width_setting:
                image = resize_image(image, max_width_setting)

            # Display a preview with alt text
            if idx % 3 == 0:
                col = col1
            elif idx % 3 == 1:
                col = col2
            else:
                col = col3

            col.image(image, caption=img_name, width=150)
            col.write(f"**GPT-4 Alt Text:** {optimized_alt_tag}")

            # Store for final export
            preview_data.append((img_name, image, optimized_alt_tag))

        # ============ Step 4: Button to Download ZIP & CSV ============
        st.markdown("---")
        st.markdown("### Download Your Optimized Files")
        if st.button("💾 Download ZIP + CSV"):
            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            zipf = zipfile.ZipFile(zip_buffer, "w")

            # Prepare CSV data
            csv_data = [
                ("Original Filename", "Optimized Alt Text", "Alt Text Length", "Exported Filename")
            ]

            for img_name, final_image, alt_tag in preview_data:
                img_bytes, exported_filename = export_image(final_image, alt_tag, user_format_choice)
                if img_bytes is None or exported_filename is None:
                    continue  # skip if there's a problem
                zipf.writestr(exported_filename, img_bytes.getvalue())

                alt_len = len(alt_tag)
                csv_data.append((img_name, alt_tag, str(alt_len), exported_filename))

            zipf.close()
            zip_buffer.seek(0)

            # Download ZIP
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

            # Download CSV
            st.download_button(
                label="📄 Download CSV Metadata",
                data=csv_bytes,
                file_name="image_metadata.csv",
                mime="text/csv"
            )

            # Show summary table
            st.markdown("#### Summary Table")
            headers = csv_data[0]
            rows = csv_data[1:]
            df = pd.DataFrame(rows, columns=headers)
            st.dataframe(df, use_container_width=True)
