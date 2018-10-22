
import os
import numpy as np
import random
import h5py

import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical



# Returns a dataset based on a list of TFRecords files passsed as a parameters. 
def create_dataset(files, batch_size=32):
    IMAGE_SHAPE = (224, 224, 3,)
    NUM_CLASSES = 6
          
    # Extract image and label from proto Example
    def _parse(example_proto):
        features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image': tf.FixedLenFeature((), tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)
        label = parsed_features['label']
        image = parsed_features['image']
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, IMAGE_SHAPE)
                                                                  
        # Pre-process image data for ResNet
        #   Substract the Imagenet mean for each channel
        imagenet_mean=np.array([103.939, 116.779, 123.68])
        image = tf.subtract(image, imagenet_mean)
        
        return image

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    dataset = dataset.repeat(count=1)

    return dataset

  
# Returns a dataset based on a list of TFRecords files passsed as a parameters. 
def create_tfrecords(bottleneck_features):
    IMAGE_SHAPE = (224, 224, 3,)
    NUM_CLASSES = 6
          
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for image_name in os.listdir(os.path.join(input_path, folder)):
            image = imread(os.path.join(input_path, folder, image_name))
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image.tobytes()),
                    'label': _int64_feature(class_to_label[folder])})
                                       
            record_writer.write(example.SerializeToString())


    return dataset


def create_bottleneck_features():
    
    # Create datset for training and validation data
    train_dataset = create_dataset(os.listdir(os.path.join(FLAGS.input_data_dir)))
    valid_dataset = create_dataset(os.listdir(os.path.join(FLAGS.input_data_dir)))

    # Create a featurizer
    featurizer = resnet50.ResNet50(
                weights = 'imagenet', 
                input_shape=(224,224,3), 
                include_top = False,
                pooling = 'avg')

    # Generate training bottleneck features
    print("Generating training bottleneck features")
    features = featurizer.predict(train_dataset, verbose=1)
    labels = retrieve_labels(train_dataset)
    
    # Save training data to TFRecords file
    filename = 'aerial_bottleneck_train.h5'
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
    filename = 'aerial_bottleneck_valid.h5'
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
tf.app.flags.DEFINE_string('file_name', 'aerial_bottleneck_train.tfrecords', 'Output file name')


def main(argv=None):
    print("Starting")
    print("Reading training data from:", FLAGS.input_data_dir)
    print("Output bottleneck files will be saved to:", FLAGS.output_data_dir)
    os.makedirs(FLAGS.output_data_dir, exist_ok=True)
   
    create_bottleneck_features()
  
if __name__ == '__main__':
    tf.app.run()