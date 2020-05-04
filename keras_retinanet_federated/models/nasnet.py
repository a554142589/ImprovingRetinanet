import keras

from keras_retinanet.models.nasnet_forked import NASNetA, ImagenetStem, load_pretrained_weights
from . import Backbone
from . import retinanet
from ..utils.image import preprocess_image


class NASNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return nasnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        return None

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['nasnet-medium', 'nasnet-mobile']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def nasnet_retinanet(num_classes, backbone='nasnet', inputs=None, modifier=None, channels=None, **kwargs):
    # choose default input
    if inputs is None:
        if channels is None:
            channels = 3

        inputs = keras.layers.Input(shape=(512, 512, channels))

    if backbone == 'nasnet-medium':
        nasnet = NASNetA(include_top=False,
                         input_tensor=inputs,
                         num_cell_repeats=6,
                         stem=ImagenetStem,
                         stem_filters=32,
                         penultimate_filters=1056,
                         num_classes=1)

        layer_names = ["conv0_bn", "concatenate_2", "concatenate_10", "concatenate_18", "concatenate_26"]
    else:
        nasnet = NASNetA(include_top=False,
                         input_tensor=inputs,
                         num_cell_repeats=4,
                         stem=ImagenetStem,
                         stem_filters=32,
                         penultimate_filters=1056,
                         num_classes=1)

        origin = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz'
        fname = 'nasnet_mobile'
        md5_hash = '7777886f3de3d733d3a6bf8b80e63555'
        load_pretrained_weights(nasnet, fname=fname, origin=origin, md5_hash=md5_hash, background_label=False)

        layer_names = ["conv0_bn", "concatenate_2", "concatenate_8", "concatenate_14", "concatenate_20"]

    if modifier:
        nasnet = modifier(nasnet)

    layer_outputs = [inputs] + [nasnet.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
