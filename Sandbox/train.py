import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.applications import resnet50
from tensorflow.keras.regularizers import l1_l2

# Returns a dataset based on a list of TFRecords files passsed as a parameters. 
def create_dataset(files, batch_size=32, prefetch_buffer_size=1, train=True, buffer_size=10000):
    IMAGE_SHAPE = (224, 224, 3,)
    NUM_CLASSES = 6
          
    # Extract image and label from proto Example
    def _parse(example_proto):
        features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image': tf.FixedLenFeature((), tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)
        label = parsed_features['label']
        label = tf.one_hot(label, NUM_CLASSES)
        image = image = tf.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, IMAGE_SHAPE)
                                                                  
        # Pre-process image data for ResNet
        #   RGB -> BGR
        image = image[..., ::-1]
        #   Substract the Imagenet mean for each channel
        imagenet_mean=tf.constant(-np.array([103.939, 116.779, 123.68]), dtype=tf.float32)
        image = tf.nn.bias_add(image, imagenet_mean)
        #image = tf.subtract(image, imagenet_mean)
        #image = resnet50.preprocess_input(image)
        
        return image, label

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse, num_parallel_calls=FLAGS.num_parallel_calls)
    #dataset = dataset.map(_parse)
    if train:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    dataset = dataset.repeat()

    return dataset

class Network():
    def __init__(self, classes=6, units=512, dropout_prob=0.5, l1=0.01, l2=0.01):
        self.base_model = resnet50.ResNet50(
            weights = 'imagenet', 
            input_shape=(224,224,3), 
            include_top = False,
            pooling = 'avg')

        x = Dense(units, activation='relu')(self.base_model.output)
        x = Dropout(dropout_prob)(x)
        y = Dense(classes, activation='softmax', kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
        self.model = Model(inputs=self.base_model.input, outputs=y)
                                                                                                
    def freeze_all_base_layers(self):
        for layer in self.base_model.layers:
            layer.trainable = False
                                                                                                                        
    def freeze_base_layers(self, num_of_layers):
        for layer in self.base_model.layers[:num_of_layers]:
            layer.trainable = False
        for layer in self.base_model.layers[num_of_layers:]:
            layer.trainable = True

    def get_model(self):
        return self.model

def get_num_of_records(files):
    count = 0
    for file in files:
        for record in tf.python_io.tf_record_iterator(file):
            count += 1
    return count


def train_evaluate():
   
    training_files = [os.path.join(FLAGS.training_files, file) for file in os.listdir(FLAGS.training_files)]
    validation_files = [os.path.join(FLAGS.validation_files, file) for file in os.listdir(FLAGS.validation_files)]
    train_dataset = create_dataset(training_files, batch_size = FLAGS.batch_size, train=True)
    valid_dataset = create_dataset(validation_files, batch_size = FLAGS.batch_size, train=False)

    fcnn = Network(units=FLAGS.units, 
                   dropout_prob = FLAGS.dropout_prob,
                   l1 = FLAGS.l1,
                   l2 = FLAGS.l2)

    fcnn.freeze_all_base_layers()
    
    model = fcnn.get_model()

    #optimizer = tf.train.AdadeltaOptimizer()
    optimizer = 'adadelta'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    steps_per_epoch = get_num_of_records(training_files)//FLAGS.batch_size
    validation_steps = get_num_of_records(validation_files)//FLAGS.batch_size

    print("Starting training:")
    print("    Epochs: ", FLAGS.epochs)
    print("    Steps per epoch: ", steps_per_epoch)
    print("    Validation steps: ", validation_steps)
    print("    Dropout prob: ", FLAGS.dropout_prob)
    print("    L1: ", FLAGS.l1)
    print("    L2: ", FLAGS.l2)

    model.fit(train_dataset,
        epochs = FLAGS.epochs,
        steps_per_epoch = steps_per_epoch,
        validation_data = valid_dataset,
        validation_steps = validation_steps)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('training_files', '', 'Location of training and validation TFRecord file')
tf.app.flags.DEFINE_string('validation_files', '', 'Location of training and validation TFRecord file')
tf.app.flags.DEFINE_integer('epochs', 5, 'Number of epochs')
tf.app.flags.DEFINE_integer('steps_per_epoch', 110, 'Number of batches per epoch')
tf.app.flags.DEFINE_integer('validation_steps', 19, 'Number of batches for validation')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of images per batch')
tf.app.flags.DEFINE_integer('prefetch_buffer_size', 1, 'Prefetch buffer size')
tf.app.flags.DEFINE_integer('units', 512, 'Number of units in a hidden layer')
tf.app.flags.DEFINE_integer('num_parallel_calls', 4,  'Parallel calls in map')
tf.app.flags.DEFINE_float('dropout_prob', 0.5,  'Dropout probability')
tf.app.flags.DEFINE_float('l1', 0.005,  'L1 regularization')
tf.app.flags.DEFINE_float('l2', 0.005,  'L2 regularization')


def main(argv=None):
   train_evaluate()


if __name__ == '__main__':
    tf.app.run()

