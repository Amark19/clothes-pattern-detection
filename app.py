from unittest import result
from flask import Flask
from flask import Flask, render_template, request
from flask import Flask, flash, request, redirect, url_for
import json
import numpy as np
from PIL import Image,ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2


##our outputs

app = Flask(__name__)
patterns = ['Checked','Woven','Embroidered','Printed','Striped']
fabrics = ['Cotton','Polyester','Nylon','Synthetic','Modal']
IMG_SIZE = 80
app.static_folder = 'static'
UPLOAD_FOLDER = '/static/'
def predicting(cnn,img):
    arr = np.array(img) 
    new_array=cv2.resize(arr,(IMG_SIZE,IMG_SIZE))
    new1_array=new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    return np.argmax(cnn.predict(new1_array))
@app.route("/",methods=[ 'POST','GET'])
def index():
    
    return render_template("index.html")
@app.route("/upload",methods=[ 'POST','GET'])
def upload():
    try:
        image =request.files['image']
        img = Image.open(image)
        img1 = ImageOps.grayscale(img)
        cnn_pt = load_model('static/pattern.model')
        cnn_fb = load_model('static/fabrics.model')
        r1 = patterns[predicting(cnn_pt,img1)]
        r2 = fabrics[predicting(cnn_fb,img1)]
        print(r1,r2)
        response=json.dumps({'r1': r1,'r2': r2})
    except:
        response=json.dumps({'r1': "error",'r2': "error"})
    return response


if __name__ == "main":
    app.run(debug=True)