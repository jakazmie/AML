
import os
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from tensorflow.keras.regularizers import l1_l2

from azureml.core import Run

import numpy as np
import random
import h5py


# Create custom callback to track accuracy measures in AML Experiment
class RunCallback(tf.keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
        
    def on_epoch_end(self, batch, logs={}):
        self.run.log(name="training_acc", value=float(logs.get('acc')))
        self.run.log(name="validation_acc", value=float(logs.get('val_acc')))


# Define network
def fcn_classifier(input_shape=(2048,), units=512, classes=6,  l1=0.01, l2=0.01):
    features = Input(shape=input_shape)
    x = Dense(units, activation='relu')(features)
    x = Dropout(0.5)(x)
    y = Dense(classes, activation='softmax', kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
    model = Model(inputs=features, outputs=y)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training regime
def train_evaluate(run):
   
    print("Loading bottleneck features")
    train_file_name = os.path.join(FLAGS.data_folder, FLAGS.train_file_name)
    valid_file_name = os.path.join(FLAGS.data_folder, FLAGS.valid_file_name)
    
    # Load bottleneck training features and labels
    with h5py.File(train_file_name, "r") as hfile:
        train_features = np.array(hfile.get('features'))
        train_labels = np.array(hfile.get('labels'))
        
        
    # Load bottleneck validation features and labels
    with h5py.File(valid_file_name, "r") as hfile:
        valid_features = np.array(hfile.get('features'))
        valid_labels = np.array(hfile.get('labels'))
        
    # Create a network
    model = fcn_classifier(input_shape=(2048,), units=FLAGS.units, l1=FLAGS.l1, l2=FLAGS.l2)
    
    run_callback = RunCallback(run)
    
    # Start training
    print("Starting training")
    model.fit(train_features, train_labels,
          batch_size=FLAGS.batch_size,
          epochs=FLAGS.epochs,
          shuffle=True,
          validation_data=(valid_features, valid_labels),
          callbacks=[run_callback])
          
    # Save the trained model to outp'uts which is a standard folder expected by AML
    print("Training completed.")
    os.makedirs('outputs', exist_ok=True)
    model_file = os.path.join('outputs', 'aerial_classifier.hd5')
    print("Saving model to: {0}".format(model_file))
    model.save(model_file)
    

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of images per batch")
tf.app.flags.DEFINE_integer('epochs', 10, "Number of epochs to train")
tf.app.flags.DEFINE_integer('units', 512, "Number of epochs to train")
tf.app.flags.DEFINE_float('l1', 0.01, "l1 regularization")
tf.app.flags.DEFINE_float('l2', 0.01, "l2 regularization")
tf.app.flags.DEFINE_string('data_folder', './bottleneck', "Folder with bottleneck features and labels")
tf.app.flags.DEFINE_string('train_file_name', 'aerial_bottleneck_train.h5', "Training file name")
tf.app.flags.DEFINE_string('valid_file_name', 'aerial_bottleneck_valid.h5', "Validation file name")

def main(argv=None):
    # get hold of the current run
    run = Run.get_submitted_run()
    train_evaluate(run)
  

if __name__ == '__main__':
    tf.app.run()