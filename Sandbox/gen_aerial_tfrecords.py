# This script processes an input folder containing aerial images into TFRecords files. The input folder structure is assumed to be:
#      Barren/
#      Cultivated/
#      Developed/
#      Herbaceous/
#      Forest/
#      Shrub/
#    
# The script will output a TFRecords file with a name passed as the script's parameter and the folloeing structure:
# 
# {'image': tf.train.Int64List,
#  'label': tf.train.BytesList}
# 
# The label encoding is as follows:
# {'Barren': 0, 'Cultivated': 1, 'Developed': 4, 'Forest': 3, 'Herbaceous': 4, 'Shrub': 5}
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import urllib
import zipfile
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import random

import tensorflow as tf

# Input parameters

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_dir', 'img_dir', "Directory with training, validation, and testing images")
tf.app.flags.DEFINE_string('out_file', 'aerial.tfrecords', "Output TFRecords file")


def convert_to_tfrecords(input_path, output_file):
  """Converts an input folder to a TFRecords file"""
  
  # Define feature encoding
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  # Iterate through images and add them to TFRecords file
  class_to_label = {
    'Barren': 0, 
    'Cultivated': 1, 
    'Developed': 4, 
    'Forest': 3, 
    'Herbaceous': 4, 
    'Shrub': 5
  } 
  
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for folder in os.listdir(input_path):
      print("Processing `{0}` images.".format(folder))
      for image_name in os.listdir(os.path.join(input_path, folder)):
        image = imread(os.path.join(input_path, folder, image_name))
        #if image.ndim < 3:
        #  image = gray2rgb(image)
        example = tf.train.Example(features=tf.train.Features(
          feature={
            'image': _bytes_feature(image.tobytes()),
            'label': _int64_feature(class_to_label[folder])
          }))
        record_writer.write(example.SerializeToString())
                            
                            
def main(arg=None):
 
  # Verify command line parameters
  if not os.path.isdir(FLAGS.img_dir): 
    print("Input directory does not exist: {0} Aborting".format(FLAGS.img_dir))
    return

  # Generate a TFRecords file
  print("Generating {0} file".format(FLAGS.out_file))
  convert_to_tfrecords(FLAGS.img_dir, FLAGS.out_file) 

if __name__ == '__main__':
  tf.app.run()

  
      
                            


