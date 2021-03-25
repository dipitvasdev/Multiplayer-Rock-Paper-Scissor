import tensorflow 
import keras_preprocessing
import numpy as np 
import cv2
model=tensorflow.keras.models.load_model("a.h5")
def predict_on_img(img_path):
    img = keras_preprocessing.image.load_img(img_path,target_size=(224,224))
    x = keras_preprocessing.image.img_to_array(img)
    x = np.expand_dims(x,axis = 0 )
    prediction = model.predict(x)
    return np.max(prediction),np.argmax(prediction)