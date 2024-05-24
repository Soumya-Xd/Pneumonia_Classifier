import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background

# Set background image
set_background('./bgs/tkt.gif')

# Custom CSS to change title, header, and classification result colors
st.markdown(
    """
    <style>
    .title {
        color: #FF6347; /* Change the title color */
    }
    .header {
        color: #1E90FF; /* Change the header color */
    }
    .result {
        color: #32CD32; /* Change the result color */
    }
    .score {
        color: #FFD700; /* Change the score color */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Set title
st.markdown('<h1 class="title">Pneumonia Classification</h1>', unsafe_allow_html=True)

# Set header
st.markdown('<h2 class="header">Please upload a chest X-ray image</h2>', unsafe_allow_html=True)

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/keras_model.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    
    # Display the uploaded image with a fixed width (500 pixels in this case)
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=False, width=500)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification result
    st.markdown(f'<h2 class="result">{class_name}</h2>', unsafe_allow_html=True)
    st.markdown(f'<h3 class="score">Score: {conf_score * 100:.1f}%</h3>', unsafe_allow_html=True)
