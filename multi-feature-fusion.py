

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=" "
import sys
import gc
import time
import cv2
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy as np
import math
import tensorflow as tf
from sklearn.utils import shuffle
import keras
from keras import *
from keras.layers import *
from keras.models import *
from keras import backend as K
import keras_resnet.models
from keras import optimizers
from keras.optimizers import SGD,Adam
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.applications import *
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.metrics import *




batchsize =64
epoc =50

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [   iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5),# vertical flips
        sometimes(iaa.Crop(percent=(0, 0.3))),# random crops
    ], random_order=True 
)

def augment(img):
    seq_det = seq.to_deterministic()
    aug_image = seq_det.augment_image(img)

    return aug_image
 
gen = ImageDataGenerator(rescale = 1./255, 
                         preprocessing_function=augment
                         #horizontal_flip=True,
                         #fill_mode='nearest')


gen_test = ImageDataGenerator(rescale=1./255)

seed = 900
train_gen1 = gen.flow_from_directory('.../rgb/train',                                                                          
                                     target_size=(224,224), 
                                     batch_size=batchsize,
                                     class_mode='binary',
                                     #shuffle=False)
                                     shuffle=True)
seed = 900
train_gen3 = gen.flow_from_directory('.../hsv/train',                                     
                                     target_size=(224,224),
                                     batch_size=batchsize,
                                     class_mode='binary',
                                     #shuffle=False)
                                     shuffle=True)
seed = 900
train_gen2 = gen.flow_from_directory('.../lbp/train',                                     
                                     target_size=(224,224),
                                     batch_size=batchsize,
                                     class_mode='binary',
                                     #shuffle=False)
                                     shuffle=True)
seed = 900
test_gen1 = gen_test.flow_from_directory('.../rgb/validation',                                         
                                         target_size=(224,224),
                                         batch_size=batchsize,
                                         class_mode='binary',
                                         #shuffle=False)
                                         shuffle=True)
seed = 900
test_gen3 = gen_test.flow_from_directory('.../hsv/validation',                                         
                                         target_size=(224,224),
                                         batch_size=batchsize,
                                         class_mode='binary',
                                         #shuffle=False)
                                         shuffle=True)
seed = 900
test_gen2 = gen_test.flow_from_directory('.../lbp/validation',                                         
                                         target_size=(224,224),
                                         batch_size=batchsize,
                                         class_mode='binary',
                                         #shuffle=False)
                                         shuffle=True)

 
def TrainGenerator(traingen1,traingen2,traingen3):
    while True:
        for (x1,y1),(x2,y2),(x3,y3) in zip(traingen1,traingen2,traingen3):            
            yield ([x1,x2,x3],y1)
            
def TestGenerator(testgen1,testgen2,testgen3):
    while True:
        for (x1,y1),(x2,y2),(x3,y3) in zip(testgen1,testgen2,testgen3):
            yield ([x1,x2,x3],y1)
            
# label_smoothing
def myloss(y_true,y_pred):
    
    alpha = 0.5   
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))   
    y0 = tf.where(tf.greater_equal(pt_0,0.5),tf.zeros_like(pt_0),pt_0)
    y0 = tf.where(tf.greater(y0,0),tf.ones_like(pt_0),tf.zeros_like(pt_0))
    y1 = tf.where(tf.greater(-pt_1,-0.5),tf.ones_like(pt_1),pt_1)
    y1 = tf.where(tf.greater(-y1,-1),tf.ones_like(pt_1),tf.zeros_like(pt_1))    
    return -K.mean(((1 - alpha) * y1 + alpha/2 )*K.log(pt_1))-K.mean(((1 - alpha) * y0 + alpha/2 )*K.log(1. - pt_0)) 

   
class GroupNormalization(Layer): 
    """
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape  

# create fusion model

conv_base = ResNet50(include_top=False,
                     input_shape=(224,224,3),
                     weights='imagenet')

conv_base2 = ResNet50(include_top=False,
                      input_shape=(224,224,3),
                      weights='imagenet')
        
conv_base3 = ResNet50(include_top=False,
                      input_shape=(224,224,3),
                      weights='imagenet')

for i, layer in enumerate(conv_base.layers):
    if "batch_normalization" in layer.name:
        conv_base.layers[i] = GroupNormalization(groups=16, axis=-1, epsilon=0.00001)
       

for i, layer in enumerate(conv_base2.layers):
    if "batch_normalization" in layer.name:
        conv_base2.layers[i] = GroupNormalization(groups=16, axis=-1, epsilon=0.00001)
        

for i, layer in enumerate(conv_base3.layers):
    if "batch_normalization" in layer.name:
        conv_base3.layers[i] = GroupNormalization(groups=16, axis=-1, epsilon=0.00001)
        



for layer in conv_base2.layers:                   
    layer.name = layer.name + str("_2")
for layer in conv_base3.layers:                   
    layer.name = layer.name + str("_3")

inp1 = conv_base.input
inp2 = conv_base2.input
inp3 = conv_base3.input

x1_low = conv_base.layers[16].output
x2_low = conv_base2.layers[16].output
x3_low = conv_base3.layers[16].output

x1_low = GlobalAveragePooling2D()(x1_low)
x2_low = GlobalAveragePooling2D()(x2_low)
x3_low = GlobalAveragePooling2D()(x3_low) 


x1 = conv_base.output
x1 = GlobalAveragePooling2D()(x1)
x1 = concatenate([x1,x1_low])

x2 = conv_base2.output
x2 = GlobalAveragePooling2D()(x2)
x2 = concatenate([x2,x2_low])

x3 = conv_base3.output 
x3 = GlobalAveragePooling2D()(x3)
x3 = concatenate([x3,x3_low])

# create AFM
x = Add()([x1,x2,x3])
e1 = Dense(units=256,activation="relu")(x)
e1 = Dense(units=2304,activation="sigmoid")(e1)    
x2 = multiply([x2, e1])
x3 = multiply([x3, e1])
x = concatenate([x1,x2,x3])

x = BatchNormalization()(x)
x = Dense(256,activation='relu',kernel_regularizer=regularizers.l1(4e-4))(x) 
x = Dense(1, activation='sigmoid')(x)
model = Model([inp1,inp2,inp3], x)


model.compile(optimizer=SGD(lr=1e-2),
              #optimizer=optimizer,
              loss=[myloss],#'binary_crossentropy',
              metrics=['accuracy'])
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50,  min_lr = 0.00000001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

# train model
history = model.fit_generator(generator=TrainGenerator(train_gen1,train_gen3,train_gen2),
                              validation_data=TestGenerator(test_gen1,test_gen3,test_gen2),                              
                              steps_per_epoch=len(train_gen1),
                              validation_steps=len(test_gen1),
                              epochs=epoc, 
                              verbose=1,
                              callbacks=[lr_reduce,early_stop]
                              )


                             
