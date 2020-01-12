from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import streamlit as st
from os import listdir
from os.path import isfile, join
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras import layers
from keras.models import load_model

IMAGE_SHAPE = (224,224)

st.sidebar.title("About")

st.sidebar.info("This Application uses a CNN (Convolutional Neural Network) trained using transfer learning to identify the dog breeds in the images.")

onlyfiles = [f for f in
             listdir("./AI_TESTING_DATA") if
             isfile(join("./AI_TESTING_DATA/",f))]

imageselect = st.sidebar.selectbox("Pick an Image.", onlyfiles)

st.title('Dog Breed Identifier')
st.write("Pick an image from the sidebar.")
st.write("Click Identify to learn the breed of the dog.")

st.write("")
image = Image.open("./AI_TESTING_DATA/" + imageselect)
st.image(image, use_column_width=True)


test_path = 'Data/Test'

model = keras.models.load_model('./models/dog_classifier_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})


if st.sidebar.button('Identify'):
    # Takes a file to test in the model
    selectedimage = "./AI_TESTING_DATA/" + imageselect
    # Opens the image and resizes the image to the correct size for the model
    selectedimage = Image.open(selectedimage).resize(IMAGE_SHAPE)

    # Places the image in an array
    selectedimage = np.array(selectedimage) / 255.0

    # Adds a batch dimension and passes the image to the model
    result = model.predict(selectedimage[np.newaxis, ...])

    # Finds the top rated probability for the prediction
    predicted_class = np.argmax(result[0], axis=-1)

    # Gets the label file from Google Drive and reads the .txt file
    labels_path = './labels/training_names.txt'
    dog_class_labels = np.array(open(labels_path).read().splitlines())

    predicted_class_name = dog_class_labels[predicted_class]

    "This is a " + predicted_class_name