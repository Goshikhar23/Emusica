from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import os
import base64
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.backend import set_session
import imutils
from keras.models import load_model



global face_model, emotion_model, emotions


emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

emotions= ["angry", "disgust", "scared", "happy", "sad", "suprised", "neutral"]


app = Flask(__name__)




def init():
    print("Loading model")

    face_model = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    emotion_model = load_model(emotion_model_path, compile=False)


@app.route("/", methods=["GET", "POST"])

def home():
    
    return render_template('home.html')



@app.route('/predict', methods=["POST", "GET"])

def emo_rec():

    if request.method == 'POST': #Capture button

        def data_uri_to_cv2_img(uri):
            encoded_data = uri.split(',')[1]
            nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            return img

        data_uri = request.form.get("data")
        gray = data_uri_to_cv2_img(data_uri)

        #gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        gray = imutils.resize(gray, width =300)
        faces = face_model.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = emotion_model.predict(roi)[0]
            label = emotions[preds.argmax()]
    
    else :
        label =None
    
    return render_template('predict.html', label = label)

    

    
            
            
    
    
            
        


if  __name__ == "__main__":
    init()
    app.debug = True
    app.run()
