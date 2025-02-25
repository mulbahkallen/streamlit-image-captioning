import streamlit as st
import openai
from PIL import Image
import io
import base64
import zipfile
import os
import csv

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
        f"1️⃣ Keep it concise (under 100 characters).\n"
        f"2️⃣ Provide context relevant to the site's content.\n"
        f"3️⃣ Naturally include keywords without keyword stuffing.\n"
        f"4️⃣ Avoid 'image of', 'picture of', or similar redundant phrases.\n"
        f"5️⃣ Provide clarity for visually impaired users.\n"
        f"Generate a final optimized alt tag now."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Function to possibly resize the image before export
def resize_image(image, max_width):
    """Optionally resize the image if its width is larger than max_width, preserving aspect ratio."""
    if image.width > max_width:
        # Calculate new height maintaining aspect ratio
        ratio = max_width / float(image.width)
        new_height = int(ratio * float(image.height))
        image = image.resize((max_width, new_height), Image.ANTIALIAS)
    return image

# Function to export image with new alt tag as filename
def export_image(image, alt_tag):
    """Save the image with the optimized alt tag as the filename (cleaned up)."""
    alt_tag_cleaned = (
        alt_tag.replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
               .replace("/", "")
               .replace("\\", "")
    )
    filename = f"{alt_tag_cleaned}.png"
    
    # Save image to a BytesIO object
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    return img_bytes, filename

# Streamlit UI
st.title("🖼️ SEO Image Alt Tag Generator")
st.markdown("""
**Welcome!**  
This tool uses GPT-4 to:
1. Generate a descriptive caption for each image.  
2. Create an SEO-optimized alt tag.  
3. Allow you to download images renamed by their alt tags in a ZIP.  
4. Optionally export a CSV with all metadata.  

*For best results, provide relevant **keywords** and a clear **theme**.*
""")

# --- Advanced Settings (Collapsible) ---
with st.expander("⚙️ Advanced Settings"):
    st.markdown("Here you can **optionally** resize images before exporting to unify their width.")
    resize_option = st.checkbox("Resize images before export?")
    if resize_option:
        max_width_setting = st.slider("Max Width (px):", min_value=100, max_value=2000, value=800, step=50)
    else:
        max_width_setting = None

# Ask user if they want to upload a single/multiple images or specify a local folder
mode = st.radio("How do you want to select images?", ["Single Image", "Multiple Images", "Folder Input"])

folder_images = []

if mode == "Folder Input":
    st.info("Enter the folder path on this machine. All .jpg, .jpeg, and .png files will be processed.")
    folder_path = st.text_input("Folder Path", value="")
    if folder_path:
        # Validate if path exists
        if os.path.isdir(folder_path):
            all_files = os.listdir(folder_path)
            # Filter for image files
            image_paths = [
                os.path.join(folder_path, f) for f in all_files
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            # Load them as "file-like" objects in memory
            for path in image_paths:
                try:
                    with open(path, "rb") as f:
                        file_bytes = f.read()
                        folder_images.append((os.path.basename(path), file_bytes))
                except Exception as e:
                    st.warning(f"Could not read file {path}: {e}")
        else:
            st.error("Invalid folder path! Please check the path and try again.")

# For single or multiple file uploads (via Streamlit interface)
if mode in ["Single Image", "Multiple Images"]:
    multiple = (mode == "Multiple Images")
    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=multiple
    )
else:
    # For folder input mode, treat images as if "uploaded" (in memory).
    uploaded_files = None

# Convert single upload to list if needed
if uploaded_files and mode == "Single Image":
    uploaded_files = [uploaded_files]

# Combine the folder-based images with the uploaded images
# We'll unify them into a list of (name, file_bytes) pairs
all_input_images = []

if uploaded_files:
    for uf in uploaded_files:
        all_input_images.append((uf.name, uf.read()))

if folder_images:
    all_input_images.extend(folder_images)

if len(all_input_images) > 0:
    st.success(f"**Total Images Found:** {len(all_input_images)}")

    # Collect or initialize session state for captions
    if "image_captions" not in st.session_state:
        st.session_state.image_captions = {}

    # Display images in columns
    col1, col2, col3 = st.columns(3)
    zip_buffer = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buffer, "w")

    # We also want to store data for CSV export
    csv_data = [("Original Filename", "Basic GPT-4 Caption", "Optimized Alt Text", "Alt Text Length", "Exported Filename")]

    for idx, (img_name, img_bytes_data) in enumerate(all_input_images):
        image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")

        # Generate caption if not already stored
        if img_name not in st.session_state.image_captions:
            with st.spinner(f"Generating GPT-4 caption for: {img_name}"):
                st.session_state.image_captions[img_name] = generate_caption_with_gpt4(image)

        # Display the image in one of three columns
        if idx % 3 == 0:
            col = col1
        elif idx % 3 == 1:
            col = col2
        else:
            col = col3

        col.image(image, caption=img_name, width=150)

    st.markdown("---")
    st.markdown("### Provide Keywords & Theme")
    st.markdown("These help GPT-4 optimize the alt text for SEO.")
    keywords_input = st.text_input("🔑 Enter target keywords (comma-separated):", "")
    theme_input = st.text_input("🎨 Enter the theme of the photos:")

    if st.button("🚀 Generate and Download"):
        if not keywords_input.strip() or not theme_input.strip():
            st.warning("Please provide both keywords and theme to proceed.")
            st.stop()

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        theme = theme_input.strip()

        for img_name, img_bytes_data in all_input_images:
            image = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
            basic_caption = st.session_state.image_captions.get(img_name, "")

            with st.spinner(f"Optimizing Alt Tag for {img_name}..."):
                optimized_alt_tag = optimize_alt_tag_gpt4(basic_caption, keywords, theme)

            alt_tag_length = len(optimized_alt_tag)

            # Optionally resize if user chose to do so
            if resize_option and max_width_setting:
                image = resize_image(image, max_width_setting)

            # Export image to memory
            img_bytes, exported_filename = export_image(image, optimized_alt_tag)

            # Write to ZIP
            zipf.writestr(exported_filename, img_bytes.getvalue())

            # Collect data for CSV
            csv_data.append((img_name, basic_caption, optimized_alt_tag, str(alt_tag_length), exported_filename))

        # Close ZIP
        zipf.close()
        zip_buffer.seek(0)

        st.markdown("---")
        st.success("All images have been processed and zipped!")
        st.download_button(
            label="📥 Download ZIP of Optimized Images",
            data=zip_buffer,
            file_name="optimized_images.zip",
            mime="application/zip"
        )

        # Create a CSV in memory
        csv_bytes = io.BytesIO()
        writer = csv.writer(csv_bytes)
        for row in csv_data:
            writer.writerow(row)
        csv_bytes.seek(0)

        st.download_button(
            label="📄 Download CSV Metadata",
            data=csv_bytes.getvalue(),
            file_name="image_metadata.csv",
            mime="text/csv"
        )

        # Display final results in a table
        st.markdown("### Summary Table")
        st.table(csv_data)  # quick way to show final data
