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
    image.save(img_bytes, format="PNG")  # for GPT vision
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
    If width is larger than max_width, resize while preserving aspect ratio.
    """
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(ratio * float(image.height))
        image = image.resize((max_width, new_height), Image.ANTIALIAS)
    return image

def export_image(image, alt_tag, user_format_choice):
    """
    Save the image using the alt_tag for the filename (cleaned).
    Respect the chosen output format (PNG, JPEG, WEBP, AVIF, etc.).
    """
    alt_tag_cleaned = (
        alt_tag.replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
               .replace("/", "")
               .replace("\\", "")
    )
    # Strip leading non-alphanumeric characters
    alt_tag_cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', alt_tag_cleaned)

    if len(alt_tag_cleaned) > 100:
        st.warning(
            f"❌ Cannot save image because filename (derived from alt tag) exceeds "
            f"100 characters:\n'{alt_tag_cleaned}'"
        )
        return None, None

    # Map user choice to extension and Pillow format
    format_mapping = {
        "PNG":  ("png",  "PNG"),
        "JPEG": ("jpg",  "JPEG"),
        "WEBP": ("webp", "WEBP"),
        "AVIF": ("avif", "AVIF")
    }
    extension, pillow_format = format_mapping[user_format_choice]

    filename = f"{alt_tag_cleaned}.{extension}"
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=pillow_format)
    img_bytes.seek(0)

    return img_bytes, filename

# ==============================
#  Streamlit UI
# ==============================
st.title("🖼️ SEO Image Alt Tag Generator")

# Track upload key in session state
if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

# Reset App Button
if st.button("Reset App"):
    if "image_captions" in st.session_state:
        del st.session_state["image_captions"]
    st.session_state["upload_key"] += 1

st.markdown("""
**Welcome!**  
1. Drag-and-drop individual images or multiple images.  
2. Alternatively, upload a **.zip folder** of images if you have a folder of images.  
3. Provide your **keywords**, **theme**, and **location** for local SEO.  
4. Generate a GPT-4 caption (internally) and SEO-optimized alt tags.  
5. Download a ZIP with renamed images **and** a CSV metadata file.
""")

# --- Advanced Settings ---
with st.expander("⚙️ Advanced Settings"):
    st.markdown("**Resize Images** (to unify widths) and choose **Output Format**.")
    resize_option = st.checkbox("Resize images before export?")
    if resize_option:
        max_width_setting = st.slider("Max Width (px):", 100, 2000, 800, 50)
    else:
        max_width_setting = None

    # Map available formats
    format_mapping = {
        "PNG":  ("png",  "PNG"),
        "JPEG": ("jpg",  "JPEG"),
        "WEBP": ("webp", "WEBP"),
        "AVIF": ("avif", "AVIF")
    }
    user_format_choice = st.selectbox(
        "Output Format for Exported Images:",
        list(format_mapping.keys()),
        index=0,
        help="Choose the final file format."
    )

# Let user choose how to provide images
upload_mode = st.radio(
    "How would you like to provide images?",
    ["Upload Images", "Upload a .zip Folder of Images"]
)

all_input_images = []

# Option 1: Upload Images
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

# Option 2: Upload a .zip
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

# If images exist, proceed
if all_input_images:
    st.success(f"**Total Images Found**: {len(all_input_images)}")

    # Initialize image_captions if not present
    if "image_captions" not in st.session_state:
        st.session_state.image_captions = {}

    # Show images in columns
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
    st.markdown(
        "These help GPT-4 optimize the alt text for SEO, including **local SEO** context."
    )
    
    keywords_input = st.text_input("🔑 Enter target keywords (comma-separated):", "")
    theme_input = st.text_input("🎨 Enter the theme of the photos:", "")
    location_input = st.text_input("📍 Enter location (for local SEO):", "")

    # Generate & Download
    if st.button("🚀 Generate & Download Alt-Optimized Images"):
        if not keywords_input.strip() or not theme_input.strip() or not location_input.strip():
            st.warning("Please provide keywords, theme, and location to proceed.")
            st.stop()

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        theme = theme_input.strip()
        location = location_input.strip()

        # Prepare ZIP
        zip_buffer = io.BytesIO()
        zipf = zipfile.ZipFile(zip_buffer, "w")

        # Prepare CSV
        csv_data = [
            ("Original Filename", "Optimized Alt Text", "Alt Text Length", "Exported Filename")
        ]

        for img_name, img_bytes_data in all_input_images:
            # 1. Generate GPT-4 caption if not cached
            if img_name not in st.session_state.image_captions:
                with st.spinner(f"Generating GPT-4 caption for {img_name}..."):
                    st.session_state.image_captions[img_name] = generate_caption_with_gpt4(img_bytes_data)

            basic_caption = st.session_state.image_captions[img_name]

            # 2. Optimize alt text
            with st.spinner(f"Optimizing Alt Tag for {img_name}..."):
                optimized_alt_tag = optimize_alt_tag_gpt4(basic_caption, keywords, theme, location)

            # 3. Resize if requested
            image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
            if resize_option and max_width_setting:
                image = resize_image(image, max_width_setting)

            # 4. Export image
            img_bytes, exported_filename = export_image(
                image, optimized_alt_tag, user_format_choice
            )
            if img_bytes is None or exported_filename is None:
                continue

            zipf.writestr(exported_filename, img_bytes.getvalue())

            # 5. Update CSV row
            alt_tag_length = len(optimized_alt_tag)
            csv_data.append((img_name, optimized_alt_tag, str(alt_tag_length), exported_filename))

        zipf.close()
        zip_buffer.seek(0)

        st.markdown("---")
        st.success("All images processed and zipped!")

        # Download ZIP
        st.download_button(
            label="📥 Download ZIP of Optimized Images",
            data=zip_buffer,
            file_name="optimized_images.zip",
            mime="application/zip"
        )

        # Create CSV in-memory
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

        # Display Summary Table
        st.markdown("### Summary Table")
        headers = csv_data[0]
        rows = csv_data[1:]
        df = pd.DataFrame(rows, columns=headers)
        st.dataframe(df, use_container_width=True)
