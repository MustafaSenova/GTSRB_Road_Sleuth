import cv2
from tensorflow import keras
import numpy as np

model = keras.models.load_model("road_sleuth.h5")

frame_width = 640
frame_height = 640
brightness = 180
threshold = 0.98
font = cv2.FONT_HERSHEY_SIMPLEX

def preprocessing(img):
    img = img / 255
    return img


cap = cv2.VideoCapture(0)
cap.set(3,frame_width)
cap.set(4,frame_height)
cap.set(10,brightness)



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
# capture the frames.
while True:
    ret, image = cap.read()
    img = np.asarray(image)
    img = cv2.resize(img,(30,30))
    img = preprocessing(img)
    img = img.reshape(1,30,30,3)
    cv2.putText(image,"Class : " , (20,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(image, "Probability : ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    probability_value = np.amax(predictions)
    if probability_value > threshold:
        print(labels[np.argmax(predictions)])
        cv2.putText(image, str(labels[np.argmax(predictions)]), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(probability_value*100,2))+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result",image)

    key = cv2.waitKey(1)
    if key == 27:  # click esc key to exit
        break


