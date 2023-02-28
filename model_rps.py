import tensorflow as tf
import numpy as np 
import cv2



interpreter = tf.lite.Interpreter("model.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

def predict_on_img(img_path):
    img = tf.keras.utils.load_img(img_path, target_size = (224,224))
    interpreter.set_tensor(input_details[0]['index'], np.float32(np.expand_dims(img, axis = 0)))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.max(output_data), np.argmax(output_data)
