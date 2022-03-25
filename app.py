import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoFeatureExtractor, VisionEncoderDecoderModel, pipeline
from PIL import Image
from huggingface_hub import InferenceApi
def predict(image, max_length=64, num_beams=4):
    image = image.convert('RGB')
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        text = tokenizer.decode(model.generate(pixel_values.cpu())[0])
        text = text.replace('<|endoftext|>', '').split('\n')
    return text[0]
model_path = "vit-gpt2-image-captioning"
device = "cpu"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
model.to(device)
print("Loaded model")
feature_extractor = AutoFeatureExtractor.from_pretrained("vit-base-patch16-224-in21k")
print("Loaded feature_extractor")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Loaded tokenizer")
title = "Image Captioning for Blind Persons"
st.title(title)
st.sidebar.title("Image Captioner")
st.sidebar.markdown("""This is a simple prototype of image captioning for blind persons. It uses GPT-2 as the language model and the VIT-Base-Patch16-224-in21k as the feature extractor. The model is trained on the VIT-Base-Patch16-224-in21k dataset. 
This is a prototype for realtime implementation more to come.
# Made with ❤️ by Vasanth. 
# [Linkedin](https://www.linkedin.com/in/vasanthengineer4949/)""")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    caption = predict(image)
    st.write(f"Caption: {caption}")
    # from gtts import gTTS
    # output = gTTS(text=caption, lang="en", slow=False)
    # output.save("output.mp3")
    # import os
    # st.audio("output.mp3")
    # os.remove("output.mp3")
