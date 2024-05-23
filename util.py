import streamlit as st
import numpy as np
from PIL import Image

def set_background(image_file):
    """
    Sets the background image for the Streamlit app.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_file});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def preprocess_image(image, target_size):
    """
    Preprocesses the input image to the target size and normalizes it.
    """
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def classify(image, model, class_names):
    """
    Classifies the input image using the provided model and class names.
    """
    target_size = (224, 224)  # Adjust according to your model's input size
    image = preprocess_image(image, target_size)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_name = class_names[class_idx]
    conf_score = prediction[0][class_idx]
    return class_name, conf_score
