#!/usr/bin/env python
import os
import gdown
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
 
app = Flask(__name__)

# Class labels
disease_classes = ['Malnutrition',
                   'Nutrition',
                 
                  ]
# Download models from Google Drive if not present
if not os.path.exists("MobileNet.h5"):
    gdown.download("https://drive.google.com/uc?id=1p7E6WrqwWdXPLNxBW7i4Ple24DhSVAf0", 
                   "MobileNet.h5", quiet=False)

if not os.path.exists("ResNet152V2.h5"):
    gdown.download("https://drive.google.com/uc?id=17HKlA0M0vOlhaTj2hf2x6p21DB5SElYM", 
                   "ResNet152V2.h5", quiet=False)


# Load models
mobilenet = load_model('MobileNet.h5')
ResNet = load_model('ResNet152V2.h5')

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    predict_x = ResNet.predict(test_image) 
    classes_x = np.argmax(predict_x, axis=1)
    return disease_classes[classes_x[0]]

def predict_labels(img_path):
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    predict_x = mobilenet.predict(test_image) 
    classes_x = np.argmax(predict_x, axis=1)
    return disease_classes[classes_x[0]]

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')
    
@app.route("/login")
def login():
    return render_template('login.html')    

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files.get('my_image')
        model = request.form.get('model')
        
        # Ensure both image and model were provided
        if img and model:
            img_path = "static/tests/" + img.filename	
            img.save(img_path)

            if model == 'ResNet152V2':
                predict_result = predict_label(img_path)
            elif model == 'MobileNet':
                predict_result = predict_labels(img_path)
            else:
                predict_result = "Unknown model selected"
                
                # Recommendation logic
            if predict_result == "Malnutrition":
                recommendation = ("⚠️ The child is identified as malnourished. "
                                  "Please consult a pediatrician immediately. A high-protein, calorie-dense diet including "
                                  "milk, eggs, legumes, and fortified foods is recommended. Ensure proper hydration, hygiene, "
                                  "and regular health monitoring.")
            else:
                recommendation = ("✅ The child is healthy. Continue providing a balanced diet rich in vegetables, fruits, grains, "
                                  "dairy, and proteins. Maintain regular growth tracking and periodic checkups.")

            return render_template("result.html", prediction=predict_result, img_path=img_path, model=model, recommendation=recommendation)
        
        return "Image or model selection is missing. Please try again.", 400
    
    return "Invalid request method. Please submit the form correctly.", 405
            
           
            
            


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
