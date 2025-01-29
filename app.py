import streamlit as st
import os
import openai
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load ViT-GPT2 model for image captioning
model_name = "Salesforce/blip-image-captioning-base"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cpu")  # Force CPU usage
model.to(device)

st.title("SEO Image Alt Tag Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def resize_image(image, max_size=(1024, 1024)):
    """Resize the image to fit within max_size while maintaining aspect ratio."""
    image.thumbnail(max_size)  # Resizes in-place while keeping aspect ratio
    return image

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    # Resize the image if it's too large
    max_size = (1024, 1024)  # Adjust this based on your memory limits
    image = resize_image(image, max_size)

    st.image(image, caption="Resized Image", use_container_width=True)

    # Generate basic caption
    with st.spinner("Generating caption..."):
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(
            pixel_values,
            max_length=16,
            num_beams=4,
            early_stopping=True
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    st.write("**Generated Caption:**", caption)

    # Get user inputs
    keywords = st.text_input("Enter target keywords (comma-separated)").split(",")
    theme = st.text_input("Enter the theme of the photos")

    if st.button("Optimize Alt Tag"):
        if not keywords or not theme:
            st.warning("Please enter both keywords and theme.")
        else:
            prompt = (
                f"Here is an image caption: '{caption}'.\n"
                f"The target keywords are: {', '.join(keywords)}.\n"
                f"The theme of the photos is: {theme}.\n"
                f"Make it SEO-friendly and descriptive."
            )

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )

            optimized_alt_tag = response["choices"][0]["message"]["content"].strip()
            st.success("Optimized Alt Tag Generated:")
            st.write(optimized_alt_tag)
