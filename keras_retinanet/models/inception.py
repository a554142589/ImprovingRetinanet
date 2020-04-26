"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import Model
from keras.layers import MaxPooling2D, AveragePooling2D, Activation, BatchNormalization, regularizers, initializers, \
    Convolution2D, concatenate
from keras_applications import imagenet_utils

from keras_retinanet.models import Backbone, retinanet
from keras_retinanet.utils.image import preprocess_image

preprocess_input = imagenet_utils.preprocess_input


def conv2d_bn(x, nb_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):
    x = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal',
                                                                      seed=None))(x)
    x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x


def block_inception_a(input, filters=8):
    branch_0 = conv2d_bn(input, filters * 3, 1, 1)

    branch_1 = conv2d_bn(input, filters * 2, 1, 1)
    branch_1 = conv2d_bn(branch_1, filters * 3, 3, 3)

    branch_2 = conv2d_bn(input, filters * 2, 1, 1)
    branch_2 = conv2d_bn(branch_2, filters * 3, 3, 3)
    branch_2 = conv2d_bn(branch_2, filters * 3, 3, 3)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, filters * 3, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x


def block_reduction_a(input, filters=8):
    branch_0 = conv2d_bn(input, filters * 3, 3, 3, strides=(2, 2))

    branch_1 = conv2d_bn(input, filters * 2, 1, 1)
    branch_1 = conv2d_bn(branch_1, filters * 2, 3, 3)
    branch_1 = conv2d_bn(branch_1, filters * 3, 3, 3, strides=(2, 2))

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=-1)
    return x


class InceptionBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return inception_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        pass

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inception']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def inception_retinanet(num_classes, backbone='inception', inputs=None, modifier=None, channels=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        if channels is None:
            channels = 3

        inputs = keras.layers.Input(shape=(None, None, channels))

    x = inputs

    # Block 1
    for i in range(4):
        x = block_inception_a(x, filters=8)

    x = block_reduction_a(x, filters=8)
    pool1 = x

    # Block 2
    for i in range(4):
        x = block_inception_a(x, filters=16)

    x = block_reduction_a(x, filters=16)
    pool2 = x

    # Block 3
    for i in range(4):
        x = block_inception_a(x, filters=32)

    x = block_reduction_a(x, filters=32)
    pool3 = x

    # Block 4
    for i in range(2):
        x = block_inception_a(x, filters=64)

    x = block_reduction_a(x, filters=64)
    pool4 = x

    # Block 5
    for i in range(2):
        x = block_inception_a(x, filters=64)

    x = block_reduction_a(x, filters=64)
    pool5 = x

    # Create model.
    inception = Model(inputs, x, name='inception')

    layer_outputs = [inputs, pool1, pool2, pool3, pool4, pool5]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
