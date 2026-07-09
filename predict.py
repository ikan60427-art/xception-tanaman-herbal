import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

MODEL_PATH = "model/xception_herbal.keras"
CLASS_PATH = "model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH,"r") as f:
    class_names = json.load(f)


def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(299,299)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    index = np.argmax(prediction)

    label = class_names[index]

    confidence = float(prediction[index])*100

    return label, confidence