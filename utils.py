 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Lim
"""

from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    """Helper function to check for valid filename."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    size = (28, 28)
    im = Image.open(image_path).resize(size=size)
    grayscale_im = im.convert('L')
    return grayscale_im

def load_model(PATH_TO_SAVED_MODEL, MODEL_NAME):
    # Load saved trained model
    try: 
        model = tf.keras.models.load_model(os.path.join(PATH_TO_SAVED_MODEL, MODEL_NAME))
    except: 
        print('Saved model not found in pretrained_model directory.')
        sys.exit()
    return model

def predict(model, image_path):
    pp_image = preprocess_image(image_path)
    pp_image.save(image_path)
    im2arr = np.array(pp_image).reshape((1, 28, 28))
    return np.argmax(model.predict(im2arr))

def remove_file(filepath):
    os.remove(filepath)
