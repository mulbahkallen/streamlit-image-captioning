import streamlit as st
import openai
from PIL import Image
import io

# Load OpenAI API key securely from Streamlit secrets
try:
    openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("🔑 OpenAI API Key is missing! Please add it in Streamlit Secrets.")
    st.stop()

# Function to get image caption using GPT-4 Vision
def generate_caption_with_gpt4(image):
    """Send image to GPT-4V and get a description."""
    
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",  # GPT-4 with Vision
        messages=[
            {"role": "system", "content": "You are an AI image captioning assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail:"},
                {"type": "image", "image": img_bytes}
            ]}
        ],
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

# Function to optimize alt text using GPT-4
def optimize_alt_tag_gpt4(caption, keywords, theme):
    """Generate an SEO-optimized alt tag using GPT-4."""
    prompt = (
        f"Here is an image caption: '{caption}'.\n"
        f"The target keywords are: {', '.join(keywords)}.\n"
        f"The theme of the photos is: {theme}.\n"
        f"Make the alt tag SEO-friendly, clear, and descriptive."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("🖼️ SEO Image Alt Tag Generator (Powered by GPT-4V)")
st.write("Upload an image to generate an AI-powered caption and optimize it for SEO!")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Generating caption with GPT-4V..."):
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
