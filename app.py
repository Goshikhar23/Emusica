from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.backend import set_session
import imutils
from keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

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
    if request.method == 'POST': #Capture button
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('images', filename))
        return redirect(url_for('prediction', filename=filename))
        
    
    return render_template('home.html')


@app.route('/prediction/<filename>', methods=["POST", "GET"])

def emo_rec():
    
    EMOTION = ""

    PHOTO = plt.imread(os.path.join('images', filename))
        
        # Preprocess image
    PHOTO = imutils.resize(PHOTO, width =300)
    gray = cv2.cvtColor(PHOTO, cv2.COLOR_BGR2GRAY)
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
    
    return render_template('predict.html', label = label)
            
        


if  __name__ == "__main__":
    init()
    app.debug = True
    app.run()