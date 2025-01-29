import streamlit as st
import openai
from PIL import Image
import io
import base64

# Load OpenAI API key securely from Streamlit secrets
try:
    openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("🔑 OpenAI API Key is missing! Please add it in Streamlit Secrets.")
    st.stop()

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
    
    # Convert image to base64
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

# Function to optimize alt text using GPT-4 Turbo
def optimize_alt_tag_gpt4(caption, keywords, theme):
    """Generate an SEO-optimized alt tag using GPT-4 Turbo."""
    prompt = (
        f"Here is an image caption: '{caption}'.\n"
        f"The target keywords are: {', '.join(keywords)}.\n"
        f"The theme of the photos is: {theme}.\n"
        f"Make the alt tag SEO-friendly, clear, and descriptive."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("🖼️ SEO Image Alt Tag Generator (Powered by GPT-4 Turbo)")
st.write("Upload an image to generate an AI-powered caption and optimize it for SEO!")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Generating caption with GPT-4 Turbo..."):
        basic_caption = generate_caption_with_gpt4(image)

    st.success("✅ Caption Generated:")
    st.write(basic_caption)

    # User input for SEO optimization
    keywords = st.text_input("🔑 Enter target keywords (comma-separated)").split(",")
    theme = st.text_input("🎨 Enter the theme of the photo")

    if st.button("🚀 Optimize Alt Tag"):
        if not keywords or not theme:
            st.warning("⚠️ Please enter both keywords and theme.")
        else:
            with st.spinner("✨ Optimizing Alt Tag..."):
                optimized_alt_tag = optimize_alt_tag_gpt4(basic_caption, keywords, theme)
            st.success("✅ Optimized Alt Tag:")
            st.write(optimized_alt_tag)
