from __future__ import division, print_function

import os

import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model("road_sleuth.h5")

threshold = 0.96


labels = {0: 'Speed limit (20km/h)',
          1: 'Speed limit (30km/h)',
          2: 'Speed limit (50km/h)',
          3: 'Speed limit (60km/h)',
          4: 'Speed limit (70km/h)',
          5: 'Speed limit (80km/h)',
          6: 'End of speed limit (80km/h)',
          7: 'Speed limit (100km/h)',
          8: 'Speed limit (120km/h)',
          9: 'No passing',
          10: 'No passing veh over 3.5 tons',
          11: 'Right-of-way at intersection',
          12: 'Priority road',
          13: 'Yield',
          14: 'Stop',
          15: 'No vehicles',
          16: 'Veh > 3.5 tons prohibited',
          17: 'No entry',
          18: 'General caution',
          19: 'Dangerous curve left',
          20: 'Dangerous curve right',
          21: 'Double curve',
          22: 'Bumpy road',
          23: 'Slippery road',
          24: 'Road narrows on the right',
          25: 'Road work',
          26: 'Traffic signals',
          27: 'Pedestrians',
          28: 'Children crossing',
          29: 'Bicycles crossing',
          30: 'Beware of ice/snow',
          31: 'Wild animals crossing',
          32: 'End speed + passing limits',
          33: 'Turn right ahead',
          34: 'Turn left ahead',
          35: 'Ahead only',
          36: 'Go straight or right',
          37: 'Go straight or left',
          38: 'Keep right',
          39: 'Keep left',
          40: 'Roundabout mandatory',
          41: 'End of no passing',
          42: 'End no passing veh > 3.5 tons'}


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (30, 30))
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 30, 30, 3)
    # PREDICT IMAGE
    predictions = model.predict(img)


    # probabilityValue =np.amax(predictions)
    return labels[np.argmax(predictions)]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)
