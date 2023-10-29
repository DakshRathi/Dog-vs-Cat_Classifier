!pip install tensorflow

import streamlit as st
import tensorflow.keras
from tensorflow.keras.models import load_model
import keras
from PIL import Image, ImageOps
import numpy as np

def machine_classification(img, weights_file):
    # Load the model
    model = tensorflow.keras.models.load_model(weights_file)

    image = img

    # image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size)

    # turn the image into a numpy array
    image = np.asarray(image)

    # Reshape the image
    image = image.reshape(1,150,150,3)

    # run the inference
    prediction = model.predict(image)

    # return prediction
    return prediction[0][0]

st.title("Dog vs Cat Image Classification")
st.write("Upload your favorite :dog:/:cat: Image for image classification as Dog or Cat")
uploaded_file = st.file_uploader("Choose a Dog/Cat jpg pic ...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded pic.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Name of the model
    label = machine_classification(image, 'model.h5')
    if label == 0.0:
        st.write("Is a :cat:")
    else:
        st.write("Is a :dog:")
