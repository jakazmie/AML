
import os
import numpy as np
import random
import h5py

import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import vgg16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# This is a generator that yields batches of preprocessed images
class ImageGenerator(tf.keras.utils.Sequence):    
    
    def __init__(self, img_dir, preprocess_fn=None, batch_size=64):
        
        # Create the dictionary that maps class names into numeric labels 
        folders = os.listdir(img_dir)
        folders.sort()
        indexes = range(len(folders))
        label_map = {key: value for (key, value) in zip(folders, indexes)}
        self.num_classes = len(label_map)
        
        # Create a list of all images in a root folder with associated numeric labels
        labeled_image_list = [(os.path.join(img_dir, folder, image), label_map[folder]) 
                              for folder in folders 
                              for image in os.listdir(os.path.join(img_dir, folder))
                              ]
        # Shuffle the list
        random.shuffle(labeled_image_list)
        # Set image list and associated label list
        self.image_list, self.label_list = zip(*labeled_image_list) 
        # Set batch size
        self.batch_size = batch_size
       
        # Set the pre-processing function passed as a parameter
        self.preprocess_fn = preprocess_fn
        
        # Set number of batches
        self.n_batches = len(self.image_list) // self.batch_size
        if len(self.image_list) % self.batch_size > 0:
            self.n_batches += 1
            
    def __len__(self):
        
        return self.n_batches
    
    def __getitem__(self, index):
        pathnames = self.image_list[index*self.batch_size:(index+1)*self.batch_size]
        images = self.__load_images(pathnames)
        
        return images
    
    # Load a set of images passed as a parameter into a NumPy array
    def __load_images(self, pathnames):
        images = []
        for pathname in pathnames:
            img = image.load_img(pathname, target_size=(224,224,3))
            img = image.img_to_array(img)
            images.append(img)
        images = np.asarray(images)
        if self.preprocess_fn != None:
            images = self.preprocess_fn(images)   
        
        return images
    
    # Return labels in one-hot encoding
    def get_labels(self):
        
        return to_categorical(np.asarray(self.label_list), self.num_classes)
    


def create_bottleneck_features():
    # Configure input directories
    train_images_dir = os.path.join(FLAGS.input_data_dir, 'train')
    valid_images_dir = os.path.join(FLAGS.input_data_dir, 'valid')

    train_generator = ImageGenerator(train_images_dir, vgg16.preprocess_input)
    valid_generator = ImageGenerator(valid_images_dir, vgg16.preprocess_input)
    
    featurizer = vgg16.VGG16(
                weights = 'imagenet', 
                input_shape=(224,224,3), 
                include_top = False,
                pooling = 'avg')
    

    # Generate training bottleneck features
    print("Generating training bottleneck features")
    features = featurizer.predict_generator(train_generator, verbose=1)
    labels = train_generator.get_labels()
    
    # Save training dataset to HDF5 file
    filename = FLAGS.training_file_name
    output_file = os.path.join(FLAGS.output_data_dir, filename)
    print("Saving training features to {}".format(output_file))
    print("   Training features: ", features.shape)
    print("   Training labels: ", labels.shape)
    with h5py.File(output_file, "w") as hfile:
        features_dset = hfile.create_dataset('features', data=features)
        labels_dset = hfile.create_dataset('labels', data=labels)

     # Generate validation bottleneck features
    print("Generating validation bottleneck features")
    features = featurizer.predict_generator(valid_generator, verbose=1)
    labels = valid_generator.get_labels()
    
    # Save validation dataset to HDF5 file
    filename = FLAGS.validation_file_name
    output_file = os.path.join(FLAGS.output_data_dir, filename)
    print("Saving validation features to {}".format(output_file))
    print("   Validation features: ", features.shape)
    print("   Validation labels: ", labels.shape)
    with h5py.File(output_file, "w") as hfile:
        features_dset = hfile.create_dataset('features', data=features)
        labels_dset = hfile.create_dataset('labels', data=labels)
    
    print("Done")

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of images per batch")
tf.app.flags.DEFINE_string('input_data_dir', 'aerialsmall', "Folder with training and validation images")
tf.app.flags.DEFINE_string('output_data_dir', 'bottleneck', "A folder for saving bottleneck features")
tf.app.flags.DEFINE_string('training_file_name', 'aerial_bottleneck_train_vgg16.h5', "Name of output training file")
tf.app.flags.DEFINE_string('validation_file_name', 'aerial_bottleneck_valid_vgg16.h5', "Name of output validation file")

def main(argv=None):
    print("Starting")
    print("Reading training data from:", FLAGS.input_data_dir)
    print("Output bottleneck files will be saved to:", FLAGS.output_data_dir)
    os.makedirs(FLAGS.output_data_dir, exist_ok=True)
   
    create_bottleneck_features()
  
if __name__ == '__main__':
    tf.app.run()