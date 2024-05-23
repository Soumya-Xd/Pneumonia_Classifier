import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background

# Set background image
set_background('./bgs/fghj.png')

# Set title
st.title('Pneumonia Classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/pneumonia_classifier.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    
    # Display the uploaded image with a fixed width (300 pixels in this case)
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=False, width=500)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification result
    st.write(f"## {class_name}")
    st.write(f"### Score: {conf_score * 100:.1f}%")
