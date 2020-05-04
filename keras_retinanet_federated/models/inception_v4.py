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
                      # kernel_regularizer=regularizers.l2(0.00004),
                      # kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
                      )(x)
    # x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x


def block_inception_a(input):
    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x


def block_reduction_a(input):
    branch_0 = conv2d_bn(input, 384, 3, 3, strides=(2, 2))

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2))

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=-1)
    return x


def block_inception_b(input):
    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x


def block_reduction_b(input):
    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2))

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, strides=(2, 2))

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=-1)
    return x


def block_inception_c(input):
    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=-1)

    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=-1)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x


class InceptionV4Backbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return inception_v4_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        pass

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inception_v4']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def inception_v4_retinanet(num_classes, backbone='inception', inputs=None, modifier=None, channels=None, **kwargs):
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

    net = conv2d_bn(inputs, 32, 3, 3, strides=(2, 2))
    net = conv2d_bn(net, 32, 3, 3)
    net = conv2d_bn(net, 64, 3, 3)

    pool1 = net

    branch_0 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net)
    branch_1 = conv2d_bn(net, 96, 3, 3, strides=(2, 2))
    net = concatenate([branch_0, branch_1], axis=-1)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3)

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    net = concatenate([branch_0, branch_1], axis=-1)
    pool2 = net

    branch_0 = conv2d_bn(net, 192, 3, 3, strides=(2, 2))
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net)

    net = concatenate([branch_0, branch_1], axis=-1)

    for idx in range(4):
        net = block_inception_a(net)

    pool3 = net

    net = block_reduction_a(net)

    for idx in range(7):
        net = block_inception_b(net)

    pool4 = net

    net = block_reduction_b(net)

    for idx in range(3):
        net = block_inception_c(net)

    pool5 = net

    layer_outputs = [inputs, pool1, pool2, pool3, pool4, pool5]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
