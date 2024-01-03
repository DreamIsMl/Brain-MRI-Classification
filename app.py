import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

mode_path = 'F:\\class\\DeepLearning\\Cnn\\Brain Tumer Classification\\model'
MODEL = tf.keras.models.load_model(mode_path)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = (256,256)
st.title('Brain Tumer Classification')

uploaded_file = st.file_uploader('Choose and image',type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)


    st.image(image, caption='Uploaded Image',use_column_width=True, width=120)
    
    if st.button('predict'):
        st.write('Processing...')
        
        image = image.resize(IMAGE_SIZE)
        image_aray = np.array(image)
        img_batch = np.expand_dims(image_aray,0)
        
        prediction = MODEL.predict(img_batch)
        preidcted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidance= np.max(prediction)
        
        st.write(f"Prediction: {preidcted_class}")
        st.write(f"Confidance: {confidance:2%}")