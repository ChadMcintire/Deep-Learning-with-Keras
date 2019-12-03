from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

def build_generator(inputs, image_size):
    """ Build a Generator Model

    Stack of BN-ReLU-Conv2DTranspose to generate fake images.
    Output activation is sigmoid instead of tahn
    Sigmoid converges easily

    #arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        image_size: Target size of one side (assuming square image)

    #Returns
        Models: Generator Model
    """

    image_resize = image_size
    # network parameters
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        # first two convolution layers use strides = 2
        # the last two use strides = 1
        if filters > layer_filters[-2]:
            strides = 2

        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator

def build_discriminator(inputs):
    """Build a Disciminator Model

    Stack of LeakyReLU-Conv2D to discriminate real from fake.
    The network does not converge with BN
    
    # Arguments
        inputs (Layer): Input layer of the disciminator (the image)

    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32,64, 128, 256]
    
    x = inputs
    for filters in layer_filters:
        #first 3 convolution layers use strides = 2
        #last one uses strides = 1

        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2

        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='disciminator')
    return discriminator

def build_and_train_models():
    #load MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()

    #reshpae data for CNN as (28, 28, 1) and normalize
    image_size = x_train.shape[1]
    x_trian = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    model_name = "dcgan_mnist"
    # network parameters
    # the latent or z vector is 100-dim
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    
    #build discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = build_discriminator(inputs)
    # [1] or original paper uses Adam,
    # but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    dicriminator.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=['accuracy'])
    discriminator.summary()

    # build generator model
    inputs_shape = (latent_size,)
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, image_size)
    generator.summary()

    #build adversarial model
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    #freeze the weights of disciminator during adversarial training
    discriminator.trainable = False
    #adversarial = generator + discriminator
    adversarial = Model(inputs,
                        disciminator(generator(inputs)),
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics['accuracy'])
    adversarial.summary()

    #train discriminator and adversarial networks
    model = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    train(models, x_train, parmas)
    
