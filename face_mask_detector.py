from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

from imutils.video import WebcamVideoStream


# url="http://192.168.2.53:8080/video"
url=0

def detectMask(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    bbox = []
    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            bbox.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            
            

            # print(len(faces))
            result = mask_detector.predict(face)
            # print(result)
            # print(len(result))
            # print("newline")
            
            
            (withoutMask) = result[0,1]
            (mask)= result[0,0]
        
            label = "check"
            print(withoutMask)
            print(mask)
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
        
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame


proto_txt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

mask_detector = load_model('mask_detector.model')


cap= WebcamVideoStream(src=url).start()

while True:
    frame = cap.read()
    if frame is None:
        print("Frame not found")
        break
    
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
   
    try:
       frame=detectMask(frame)
    except:
       print("Something went wrong")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()