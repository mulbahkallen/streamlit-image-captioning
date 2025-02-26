import streamlit as st
import openai
from PIL import Image
import io
import base64
import zipfile
import os
import csv
import pandas as pd
import re  # <-- NEW for regex

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

def export_image(image, alt_tag, output_format):
    """
    Save the image with the optimized alt tag as the filename.
    Also respect the chosen output format (png/jpg).
    """

    alt_tag_cleaned = (
        alt_tag.replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
               .replace("/", "")
               .replace("\\", "")
    )

    # ---------- NEW: Strip leading special characters ----------
    alt_tag_cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', alt_tag_cleaned)
    # -----------------------------------------------------------

    # ---------- NEW: Enforce 100-character limit ---------------
    if len(alt_tag_cleaned) > 100:
        st.warning(
            f"❌ Cannot save image because filename (derived from alt tag) exceeds "
            f"100 characters:\n'{alt_tag_cleaned}'"
        )
        return None, None
    # -----------------------------------------------------------

    extension = "png" if output_format == "PNG" else "jpg"
    filename = f"{alt_tag_cleaned}.{extension}"

    # Save the image to a BytesIO buffer
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=output_format)
    img_bytes.seek(0)

    return img_bytes, filename

# ==============================
#  Streamlit UI
# ==============================

st.title("🖼️ SEO Image Alt Tag Generator")

# --------- NEW: Reset Button -----------
if st.button("Reset App"):
    st.session_state.clear()
    st.experimental_rerun()
# ---------------------------------------

st.markdown("""
**Welcome!**  
1. Drag-and-drop individual images or multiple images.  
2. Alternatively, upload a **.zip folder** of images (drag-and-drop) if you have a folder of images.  
3. Provide your **keywords**, **theme**, and now **location** for local SEO.  
4. Generate a GPT-4 caption (internally) and SEO-optimized alt tags.  
5. Download a ZIP with renamed images **and** a CSV metadata file (no GPT-4 caption displayed).
""")

# --- Advanced Settings (Collapsible) ---
with st.expander("⚙️ Advanced Settings"):
    st.markdown("**Resize Images** (to unify widths) and choose **Output Format**.")
    resize_option = st.checkbox("Resize images before export?")
    if resize_option:
        max_width_setting = st.slider("Max Width (px):", min_value=100, max_value=2000, value=800, step=50)
    else:
        max_width_setting = None
    
    output_format = st.selectbox(
        "Output Format for Exported Images:",
        ["PNG", "JPEG"], 
        index=0,
        help="Choose the final file format (PNG or JPEG)."
    )

# Let user choose whether they want to upload files or a zip folder
upload_mode = st.radio(
    "How would you like to provide images?",
    ["Upload Images", "Upload a .zip Folder of Images"]
)

all_input_images = []

# ========== Option 1: Upload Images ==========
if upload_mode == "Upload Images":
    uploaded_files = st.file_uploader(
        "Drag and drop or select images",
        type=["jpg", "jpeg", "png", "webp", "avif"],  # NEW: accept webp + avif
        accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            all_input_images.append((uf.name, uf.read()))

# ========== Option 2: Upload a .zip ==========
elif upload_mode == "Upload a .zip Folder of Images":
    uploaded_zip = st.file_uploader(
        "Drag and drop or select a .zip file of images",
        type=["zip"],
        accept_multiple_files=False
    )
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    # Only process files with valid image extensions
                    filename_lower = file_info.filename.lower()
                    if filename_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):  # NEW
                        file_bytes = zip_ref.read(file_info.filename)
                        base_name = os.path.basename(file_info.filename)
                        all_input_images.append((base_name, file_bytes))

if all_input_images:
    st.success(f"**Total Images Found**: {len(all_input_images)}")

    # Collect or initialize session state for GPT-4 captions
    if "image_captions" not in st.session_state:
        st.session_state.image_captions = {}

    # Show the images in columns
    col1, col2, col3 = st.columns(3)
    for idx, (img_name, img_bytes_data) in enumerate(all_input_images):
        image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")

        if idx % 3 == 0:
            col = col1
        elif idx % 3 == 1:
            col = col2
        else:
            col = col3

        col.image(image, caption=img_name, width=150)

    st.markdown("---")
    st.markdown("### Provide Keywords, Theme & Location")
    st.markdown("These help GPT-4 optimize the alt text for SEO, including **local SEO** context.")
    keywords_input = st.text_input("🔑 Enter target keywords (comma-separated):", "")
    theme_input = st.text_input("🎨 Enter the theme of the photos:", "")
    location_input = st.text_input("📍 Enter location (for local SEO):", "")

    if st.button("🚀 Generate & Download Alt-Optimized Images"):
        if not keywords_input.strip() or not theme_input.strip() or not location_input.strip():
            st.warning("Please provide keywords, theme, and location to proceed.")
            st.stop()

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        theme = theme_input.strip()
        location = location_input.strip()

        # Prepare a ZIP to store final images
        zip_buffer = io.BytesIO()
        zipf = zipfile.ZipFile(zip_buffer, "w")

        # Prepare CSV data (NO GPT-4 caption displayed)
        csv_data = [
            ("Original Filename",
             "Optimized Alt Text",
             "Alt Text Length",
             "Exported Filename")
        ]

        for img_name, img_bytes_data in all_input_images:
            # 1. Generate GPT-4 basic caption (cached) for alt text optimization
            if img_name not in st.session_state.image_captions:
                with st.spinner(f"Generating GPT-4 caption for {img_name}..."):
                    st.session_state.image_captions[img_name] = generate_caption_with_gpt4(img_bytes_data)

            basic_caption = st.session_state.image_captions[img_name]

            # 2. Optimize alt text
            with st.spinner(f"Optimizing Alt Tag for {img_name}..."):
                optimized_alt_tag = optimize_alt_tag_gpt4(basic_caption, keywords, theme, location)

            # 3. Possibly resize
            image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
            if resize_option and max_width_setting:
                image = resize_image(image, max_width_setting)

            # 4. Export with alt-tag-based filename
            img_bytes, exported_filename = export_image(image, optimized_alt_tag, output_format)
            
            # If export_image returned None, skip writing to zip & CSV
            if img_bytes is None or exported_filename is None:
                continue

            zipf.writestr(exported_filename, img_bytes.getvalue())

            # 5. Collect row data for CSV
            alt_tag_length = len(optimized_alt_tag)
            csv_data.append((img_name, optimized_alt_tag, str(alt_tag_length), exported_filename))

        # Close ZIP
        zipf.close()
        zip_buffer.seek(0)

        st.markdown("---")
        st.success("All images have been processed and zipped!")

        # ---- Download ZIP
        st.download_button(
            label="📥 Download ZIP of Optimized Images",
            data=zip_buffer,
            file_name="optimized_images.zip",
            mime="application/zip"
        )

        # CREATE CSV in-memory
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

        # ---- Display Summary Table (without GPT-4 caption)
        st.markdown("### Summary Table")
        headers = csv_data[0]
        rows = csv_data[1:]  # all data except header
        df = pd.DataFrame(rows, columns=headers)
        st.dataframe(df, use_container_width=True)
