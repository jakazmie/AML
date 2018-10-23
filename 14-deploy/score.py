import json
import os
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications import vgg16

from azureml.core.model import Model

def init():
    # Instantiate VGG16 featurizer
    global featurizer
    featurizer = vgg16.VGG16(
            weights = 'imagenet', 
            input_shape=(224,224,3), 
            include_top = False,
            pooling = 'avg')

    # Load the model
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('aerial_sklearn')
    model = joblib.load(model_path)
  

def run(raw_data):
    # convert json to numpy array
    img = np.array(json.loads(raw_data)['data'])
    # normalize as required by ResNet50
    img = vgg16.preprocess_input(img)
    # extract bottleneck features
    features = featurizer.predict(img)
    # make prediction
    predictions = model.predict(features)
    return json.dumps(predictions.tolist())