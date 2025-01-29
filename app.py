import streamlit as st
import os
import openai
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load ViT-GPT2 model for image captioning
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("SEO Image Alt Tag Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

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
