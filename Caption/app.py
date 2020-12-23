from __future__ import division, print_function

import pickle
from flask import Flask, render_template, request, flash
import numpy as np
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
# from pickle import load
from PIL import Image

# app = Flask(__name__)

UPLOAD_FOLDER = './Web/uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/uploads',
            static_folder='./Web/uploads',
            template_folder='./Web')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/index.html')
def uploadCap():
   return render_template('index.html')

def extract_features(filename, model):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

@app.route('/uploaded_cap', methods = ['POST'])
def uploaded_cap():
    if request.method == 'POST':

        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
        # img_path= './Web/uploads/images/upload_caps.jpg'
        max_length = 32
        tokenizer = pickle.load(open("tokenizer.p","rb"))
        model = load_model('models/model_9.h5')
        xception_model = Xception(include_top=False, pooling="avg")

        photo = extract_features(file_path, xception_model)

        description = generate_desc(model, tokenizer, photo, max_length)
        description=description.replace('start','')
        description=description.replace('end','')
        description = description[1].upper() + description[2:]
        P='uploads/images/'+filename
        return render_template('index.html',d=description, C=P)

       # https://github.com/Uttam580?tab=repositories
if __name__=="__main__":
    app.run(debug=True)
