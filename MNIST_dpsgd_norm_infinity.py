from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import matplotlib.pyplot as plt
from keras import backend as K
import scipy.io as io
import os
import pickle


mirrored_strategy = tf.distribute.MirroredStrategy()

###########################################################################
# System Parameters
###########################################################################
tf.flags.DEFINE_float('learning_rate', .3, 'Learning rate for training')
tf.flags.DEFINE_integer('batch_size', 10000, 'Batch size')
tf.flags.DEFINE_integer('epochs', 200, 'Number of epochs')
tf.flags.DEFINE_float('norm_clip', 1/100, 'clip l2 of the gradient')
tf.flags.DEFINE_float('eps0', 1.5, 'central privacy level epsilon')
tf.flags.DEFINE_float('delta', 1e-5, 'delat')
FLAGS = tf.flags.FLAGS
num_samples = 60000

###########################################################################
# Load MNIST Dataset
###########################################################################
def load_mnist():
  img_rows, img_cols = 28, 28
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  y_train = y_train.astype('int32')
  y_test = y_test.astype('int32')
  assert x_train.min() == 0.
  assert x_train.max() == 1.
  assert x_test.min() == 0.
  assert x_test.max() == 1.
  assert y_train.ndim == 1
  assert y_test.ndim == 1
  train_dataset = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(y_train, tf.int32)))
  train_dataset = train_dataset.shuffle(buffer_size=1000).batch(FLAGS.batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(x_test, tf.float32), tf.dtypes.cast(y_test, tf.int32)))
  test_dataset = test_dataset.shuffle(buffer_size=1000).batch(10000)
  return train_dataset, test_dataset, input_shape
###########################################################################
# Define Model 
###########################################################################
class MyModel(tf.keras.Model):
    def __init__(self,input_shap):
        super(MyModel, self).__init__()
        self.input_shap = input_shap
        self.conv_1 = tf.keras.layers.Conv2D(16,8, strides=2, padding='same', activation='tanh', input_shape=self.input_shap)
        self.conv_2 = tf.keras.layers.Conv2D(16,4, strides=2, padding='valid', activation='tanh')
        self.maxp_1 = tf.keras.layers.MaxPool2D(2,1)
        self.maxp_2 = tf.keras.layers.MaxPool2D(2,1)
        self.dense1 = tf.keras.layers.Dense(32, activation= 'tanh')
        self.dense2 = tf.keras.layers.Dense(10)
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.maxp_1(x)
        x = self.conv_2(x)
        x = self.maxp_2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
def my_vector_loss(y_true, y_pred):
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
  return tf.reduce_mean(vector_loss)
############################################################################
# FLatten a list of gradients
############################################################################
def Flatten_gradients(gradients):
    shape_of_gradients =  [x.shape for x in gradients]
    return tf.concat([K.flatten(x) for x in gradients], axis=0), shape_of_gradients 
############################################################################
# Infinity mechanism : infty-Norm Clip the gradient and add noise
############################################################################
def Private_gradients(gradient):
    "choose at random one coordinate to quantize"
    D_len = len(gradient)
    Z = int(np.random.choice(D_len,1)) 
    "clip the gradient"
    norm = tf.math.abs(gradient[Z])
    grads = gradient[Z] /tf.maximum(norm/FLAGS.norm_clip,1)
    "LDP quantization of a single coordinate" 
    C = (np.exp(FLAGS.eps0)+1)/(np.exp(FLAGS.eps0)-1)
    pr = 0.5 + (grads/(2*C*FLAGS.norm_clip))
    x = 2*np.random.binomial(1,pr,1)-1
    "Dequantizater at the aggregator"
    H =np.zeros_like(gradient)
    H[Z] = D_len*x*C*FLAGS.norm_clip
    return H
############################################################################
# Return the shape of gradients
############################################################################
def reshape_gradients(gradient, shape_of_gradients):
    grads = []
    len_current = 0
    for i in range(0,len(shape_of_gradients)):
        len_of_grad = tf.math.reduce_prod(shape_of_gradients[i])
        grads.append(tf.reshape(gradient[len_current:len_current+len_of_grad],shape_of_gradients[i]))
        len_current += len_of_grad
    return grads
############################################################################
# Change Learning rate per epoch
############################################################################
def lr_schedule (epoch):
    if epoch > 70:
        return 0.18
    else:
        return FLAGS.learning_rate
###########################################################################
# Design Training loop from scratch
###########################################################################
def train(epoch, train_dataset):
    print('\nEpoch: %d' % epoch)
    progbar = tf.keras.utils.Progbar((len(list(train_dataset.as_numpy_iterator()))))
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataset):
        total += len(inputs)
        batch_loss = 0
        batch_acc = 0
        for i in range(0,len(inputs)):
            with tf.GradientTape(persistent=True) as tape:
                outputs = model(tf.expand_dims(inputs[i], axis=0), training=True)
                loss = my_vector_loss(tf.expand_dims(targets[i], axis=0), outputs) 
            grads = tape.gradient(loss, model.trainable_weights)
            Flatten_grads, shape_of_grads = Flatten_gradients(grads)
            Private_grads = Private_gradients(Flatten_grads)
            if i==0:
                Total_grads = Private_grads
            else:
                Total_grads += Private_grads
            batch_loss += float(loss)
            train_acc_metric.update_state(targets[i], outputs)
            batch_acc += float(train_acc_metric.result())
            train_acc_metric.reset_states()
        Total_grads /= len(inputs)
        reshape_grads = reshape_gradients(Total_grads, shape_of_grads)
        optimizer.lr.assign(lr_schedule(epoch))   
        optimizer.apply_gradients(zip(reshape_grads, model.trainable_weights))
        batch_loss /= len(inputs)
        train_loss += float(batch_loss)
        correct += float(batch_acc)
        progbar.update(batch_idx+1, values=[("train loss", train_loss/(batch_idx+1)), ("acc", correct/(total))] )
    train_acc[epoch-1] = correct/(total)
###########################################################################
# Design Testing loop from scratch
###########################################################################
def test(epoch, test_dataset):
    progbar = tf.keras.utils.Progbar((len(list(test_dataset.as_numpy_iterator()))))
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_dataset):
        total += len(inputs)
        outputs = model(inputs, training=False)
        test_loss += my_vector_loss(targets, outputs) 
        test_acc_metric.update_state(targets, outputs)
        correct += len(inputs)* float(test_acc_metric.result())
        test_acc_metric.reset_states()
        progbar.update(batch_idx+1, values=[("test loss", test_loss/(batch_idx+1)), ("acc", float(correct)/total)] )
    test_acc[epoch-1] = float(correct)/total
###########################################################################
# Main run
###########################################################################
train_data, test_data, input_shape = load_mnist()
model = MyModel(input_shape)
train_acc = np.zeros((FLAGS.epochs,))
test_acc = np.zeros((FLAGS.epochs,))
total_eps = np.zeros((FLAGS.epochs,))
############################################################################
# Loss function and optimizer
############################################################################
optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
###########################################################################  
filename = 'data_'+'eps_mnist_l3_10'    
for epoch in range(1, FLAGS.epochs+1):
    total_eps[epoch-1] = compute_privacy(num_samples,FLAGS.batch_size,epst,epoch,FLAGS.delta)
    train(epoch, train_data)
    test(epoch, test_data)
    data = {
        'test_acc': test_acc,
        'train_acc' : train_acc,
        'Eps' : total_eps,
        'batch_size': FLAGS.batch_size,
        'eps0': FLAGS.eps0
    }
    io.savemat(filename,data)