import streamlit as st
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import openai

# Load OpenAI API key securely from Streamlit secrets
try:
    openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("🔑 OpenAI API Key is missing! Please add it in Streamlit Secrets.")
    st.stop()

# Load BLIP-2 Model (More Accurate Than BLIP-1)
@st.cache_resource
def load_blip2_model():
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)  # Uses float16 for efficiency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_blip2_model()

# Function to resize large images (reduces memory usage)
def resize_image(image, max_size=(1024, 1024)):
    """Resize the image to fit within max_size while maintaining aspect ratio."""
    image.thumbnail(max_size)
    return image

# Function to generate captions using BLIP-2
def generate_caption(image):
    """Generate a caption for an image using the BLIP-2 model."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs)

    caption = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# Function to optimize alt text using GPT-4
def optimize_alt_tag_gpt4(caption, keywords, theme):
    """Generate an SEO-optimized alt tag using GPT-4."""
    prompt = (
        f"Here is an image caption: '{caption}'.\n"
        f"The target keywords are: {', '.join(keywords)}.\n"
        f"The theme of the photos is: {theme}.\n"
        f"Make the alt tag SEO-friendly, clear, and descriptive."
    )

    response = openai_client.chat.completions.create(  # NEW: Updated API call
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("🖼️ SEO Image Alt Tag Generator (BLIP-2)")
st.write("Upload an image to generate an AI-powered caption and optimize it for SEO!")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    # Resize if too large
    image = resize_image(image)

    st.image(image, caption="Resized Image", use_container_width=True)

    with st.spinner("🔍 Generating caption..."):
        basic_caption = generate_caption(image)

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
