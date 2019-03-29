 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Lim
"""

from __future__ import absolute_import, division, print_function
import os
import glob
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, flash, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from utils import allowed_file, preprocess_image, load_model, predict, remove_file

# Setup paths and directoriess
PATH_TO_SAVED_MODEL = './pretrained_model'
MODEL_NAME = 'mnist_model.h5'

# Instantiate Flask application and configuration parameters
app = Flask(__name__)
app.config['SECRET_KEY'] = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = './tmp'
app.config['MAX_CONTENT_LENGTH'] = 16*1026*1024   # 16MB max file size

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def upload_test_image():
    if request.method == 'POST':
        # check if post request has the file
        if 'file' not in request.files:
            flash('No file.')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also 
        # submits empty part without filename
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)
        # valid file uploaded by user
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            model = load_model(PATH_TO_SAVED_MODEL, MODEL_NAME)
            prediction = predict(model, filepath)
            return render_template('results.html', prediction=prediction, filename=filename)
    # clear /tmp folder
    for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        os.remove(f)
    return render_template('index.html')

@app.route('/tmp/<filename>', methods=['GET', 'POST'])
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()

