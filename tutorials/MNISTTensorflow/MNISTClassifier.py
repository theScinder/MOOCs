""" cNN based on tutorial from https://www.tensorflow.org/tutorials/layers"""
# imports associated with tensorflow tutorial: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import tensorflow as tf
import time # 

tf.logging.set_verbosity(tf.logging.INFO)
# Essential Imports
# math etc.
from scipy import misc
import numpy as np
#import scipy as sp

#plotting
import matplotlib as mpl
from matplotlib import pyplot as plt

#image functions, esp. resizing
import cv2
#directory functions
import os

#for import .mat files
from scipy import io

# Describe the network architecture
# The layers will be input -> reLu(conv1) -> pooling1 -> reLU(conv2) -> pooling2 -> hiddenLayer > logitsLayer
# Input images are 100x100 px FOVs from VE-BF or TIE microscopy of microtubules
# reference: https://www.tensorflow.org/tutorials/layers

# Graph parameters
convDepth = 16
imgHeight = 28
imgWidth = 28
pool1Size = 5
pool2Size = 4


nVisible = imgHeight * imgWidth
nHiddenDense = 1024
nFlatPool = round(imgHeight/pool1Size/pool2Size)

# learning parameters
lR = 1e-3 # learning rate
mom = 1e-2 #momentum
myLambda = 1e-50#1e-13 # weight penalty
batchSize = 79
dispIt = 3 # display every th epoch
epochs = 9 # train through full dataset for this many epochs
logIter = 500

        
def cNNMTModel(data, labels, mode):
    # mode is a boolean that determines whether to apply dropout (for training)
    # or keep all layers (evaluation/test data)
    #inputLayer = tf.reshape(data, [-1,100,100,1])
    inputLayer = tf.reshape(data, [-1, 28, 28, 1])

    # First convolutional layer and pooling 
    conv1 = tf.layers.conv2d(
        inputs = inputLayer,
        filters = convDepth,
        kernel_size = [10,10],
        padding = "same",
        activation = tf.nn.relu)
    # pooling (reduce size for faster learning)
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = pool1Size,
        strides = pool1Size)
    # Second convo layer and pooling
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters = convDepth*2,
        kernel_size = [10,10],
        padding = "same",
        activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = pool2Size, # for square pool sizes can specify single number as size
        strides = pool2Size)
    
    # dense layer
    # 5x5 depends on the max pooling. Here I've using max_pool2d with 
    # pool sizes of 5 and 4, so the dimension should be 100/5/4 = 5
    #
    pool2Flat = tf.reshape(pool2,
                           [-1,
                            nFlatPool*nFlatPool*convDepth*2])
    denseHL = tf.layers.dense(inputs=pool2Flat,
                             units=nHiddenDense,
                             activation=tf.nn.relu)
    
    # If mode is True, than apply dropout (training)
    dropout = tf.layers.dropout(inputs=denseHL,
                               rate=0.1,
                               training = mode == learn.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout,
                             units=10) # 2 units for 2 classes: w & w/o MTs
    

    # loss and training op are None
    loss = None
    #trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss)
    trainOp = None
    
    # Loss for TRAIN and EVAL modes
    if mode != learn.ModeKeys.INFER:
        oneHotLabels = tf.one_hot(indices = tf.cast(labels,tf.int32),
                                 depth=10) 
        
        # Because my labels are already one hot (not indexes), don't have to call tf.one_hot
        #oneHotLabels = labels#nm
        loss = tf.losses.softmax_cross_entropy(onehot_labels = oneHotLabels,
                                              logits = logits)
        
        #tf.summary.scalar('cross_entropy', loss)

    # Training op
    if mode == learn.ModeKeys.TRAIN:
        trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss,global_step = tf.contrib.framework.get_global_step())
        #trainOp = tf.contrib.layers.optimize_loss(
        #loss = loss,
        #global_step = tf.contrib.framework.get_global_step(),
        #learning_rate = lR,
        #optimizer = "SGD")
    
    # Gen. Pred.
    predictions = {
        "classes": tf.argmax(
        input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name = "softmaxTensor")
    }
    
    # attach summaries for tensorboad https://www.tensorflow.org/get_started/summaries_and_tensorboard

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss, train_op=trainOp)


def main(unused_argv):
    # Load the training data
   
    #7392 in training data
    mnist = learn.datasets.load_dataset("mnist")
    trainData = mnist.train.images # Returns np.array
    trainLabels = np.asarray(mnist.train.labels, dtype=np.int32)
    evalData = mnist.test.images # Returns np.array
    evalLabels = np.asarray(mnist.test.labels, dtype=np.int32)

    print("labels shape (training): ", np.shape(trainLabels)," (evaluation): ", np.shape(evalLabels))
    print("mean value for evaluation labels (coin-flip score): ", np.mean(evalLabels))
   
    print(trainData[0:20])
    
    print("labels shape (training): ", np.shape(trainLabels)," (evaluation): ", np.shape(evalLabels))
    print("mean value for evaluation labels (coin-flip score): ", np.mean(evalLabels))
    sTime = time.time()
    # Create estimator
    MTClassifier = learn.Estimator(model_fn = cNNMTModel,
                                   model_dir = "./MNIST/MTConvNetModel")
    # set up logging
    tensors_to_log = {"probabilities": "softmaxTensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,
                                              every_n_iter = 100)
    
    
    # Train Model 
    MTClassifier.fit(x=trainData,
                    y=trainLabels,
                    batch_size = batchSize,
                    steps = 10000,
                    monitors = [logging_hook])
    
    # Metrics for evaluation
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                           prediction_key="classes")
              }
    print(np.mean(evalLabels))
    print("elapsed time: ",time.time()-sTime)
    # Evaluate model and display results
    evalResults = MTClassifier.evaluate(x=evalData,
                                          y=evalLabels,
                                          metrics=metrics)
    print("wobobobob", evalResults)
    print(np.mean(trainData))
    print(np.mean(evalData))



if __name__ == "__main__":
    tf.app.run()


