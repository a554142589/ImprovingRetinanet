#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import os
import random
import sys
import warnings
import time 

import collections
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image
import tensorflow_federated as tff
from tensorflow.python.keras.optimizer_v2 import gradient_descent
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_v2_behavior()
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.group_csv_generator import GroupCSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.data_loader import _load_data_test
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator

import numpy as np



def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        print("Creating model with weights...")
        model.load_weights(weights, by_name=True)
    return model


def create_models(args, backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, config=None, model_config=None):
    """ Creates three models (model, training_model, prediction_model).
    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    if model_config is None:
        model_config = dict()

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    channels = None
    if config and 'dimensions' in config:
        channels = int(config['dimensions']['channels'])

    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, channels=channels, model_config=model_config),
                                       weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, channels=channels, model_config=model_config), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    # prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    def loss_fn(y_true, y_pred):
        return tf.keras.losses.MSE(y_true, y_pred)
 
    training_model.compile(
        loss=[losses.smooth_l1_loss, losses.mask_focal_loss],#,losses.mask_focal_loss,losses.mask_focal_loss],
            # 'regression': losses.smooth_l1(),
            # 'classification': losses.focal(),
            # 'mask': loss_fn#'mean_squared_error',
            # 'mask': losses.mask_focal(),
    
        optimizer=keras.optimizers.Adam(lr=args.lr, clipnorm=0.001), # used to be 1e-4

    )
    return training_model

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval
            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average, max_detections=args.max_detections)

        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    # if args.snapshots:
    if args.snapshot_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        print('saving models to path: ', args.snapshot_path)
        print('save best only:', args.best_only)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(args.snapshot_path, '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)),
            verbose=1,
            save_best_only=args.best_only,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    print('Reduce LR:', args.reduce_lr)
    if args.reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='mAP',
            factor=0.1,
            patience=2,
            verbose=1,
            mode='max',
            min_delta=0.0001,
            cooldown=3,
            min_lr=0
        ))

    # callbacks.append(keras.callbacks.EarlyStopping(
    #     monitor='mAP',
    #     min_delta=0,
    #     patience=4,
    #     verbose=1,
    #     mode='max',
    #     restore_best_weights=True
    # ))

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'config': args.config,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
        'preprocess_image': preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            **common_args
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            **common_args
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'group_csv':
        train_generator = GroupCSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = GroupCSVGenerator(
                args.val_annotations,
                args.classes,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            args.main_dir,
            subset='train',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = OpenImagesGenerator(
            args.main_dir,
            subset='validation',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        train_generator = KittiGenerator(
            args.kitti_path,
            subset='train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            args.kitti_path,
            subset='val',
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn(
            'Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type_dummy')
    # subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version', help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter', help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    ## Here is for CT DeepLesion
    # csv_parser.add_argument('annotations', default='results_formatted_2mm.csv', help='Path to CSV file containing annotations for training.')
    # csv_parser.add_argument('classes', default='mapping.csv', help='Path to a CSV file containing class label mapping.')
    # csv_parser.add_argument('--val-annotations', default='validation_formatted_2mm.csv', help='Path to CSV file containing annotations for validation (optional).')
    ## Here is for Whole-body MRI
    csv_parser.add_argument('annotations', default='results.csv', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', default='mapping.csv', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', default='results.csv', help='Path to CSV file containing annotations for validation (optional).')

    group_csv_parser = subparsers.add_parser('group_csv')
    group_csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    group_csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    group_csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', default=None, help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights', help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights', default=None, help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone', default='vgg19', help='Backbone model used by retinanet.', type=str)
    parser.add_argument('--batch-size', default=4, help='Size of the batches.', type=int)
    parser.add_argument('--gpu', default='1', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')

    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100) # used to be 50
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=100) # used to be 10000
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='../snapshots')
    parser.add_argument('--reduce_lr', help='ReduceLROnPlateau', action='store_true')

    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='../logs')
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', default=False, help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', default=False, help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', default='/research/dept6/yhlong/mrJiang/Improveing_RetinaNet_Code/code/deeplesion/config/anchors4.ini',
                        help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--best_only', help='Only save the best model.', action='store_true')
    parser.add_argument('--max-detections', help='Number of detections to consider during evaluation.', type=int, default=100)
    parser.add_argument('--model-config', help='Path to a model configuration.', type=str)

    parser.add_argument('--dataset_type', default='csv', help='Arguments for specific dataset types')
    parser.add_argument('--classes', default='mapping.csv', help='Path to a CSV file containing class label mapping.')
    # parser.add_argument('--annotations', default='results_formatted_2mm.csv', help='Path to CSV file containing annotations for training.')
    # parser.add_argument('--val-annotations', default='validation_formatted_2mm.csv', help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--annotations', default='results.csv', help='Path to CSV file containing annotations for training.')
    parser.add_argument('--val-annotations', default='results.csv', help='Path to CSV file containing annotations for validation (optional).')

    return check_args(parser.parse_args(args))


def main(args=None, model_config=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Set empty config
    if model_config is None:
        if args.model_config:
            model_config = json.load(open(args.model_config))[0]
        else:
            model_config = {}

    # seed
    np.random.seed(42)
    tf.set_random_seed(42)
    random.seed(42)
    # sess = get_session()
    gpu_options = tf.GPUOptions(allow_growth=False)
    config_proto = tf.ConfigProto(gpu_options=gpu_options)
    # off = rewriter_config_pb2.RewriterConfig.OFF

    # config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    # g = tf.Graph()
    # keras.backend.set_session(sess)

    # with g.as_default():
        # create object that stores backbone information

    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    # anno_base_dir = '/vol/medic02/users/qdou20/projects/lesion_detection/clean_code/ct_csv_files'
    # anno_base_dir = '/vol/medic02/users/qdou20/projects/lesion_detection/clean_code/mr_csv_files'
    # anno_base_dir = '/research/dept6/yhlong/mrJiang/DeepLesionDataSet/'
    anno_base_dir = '/research/dept6/yhlong/mrJiang/COVIDDataSet/'
    # anno_base_dir = '/Users/jemary/Data/Code/Improving_retina/code/CONVID/'
    args.annotations = os.path.join(anno_base_dir, args.annotations)
    args.classes = os.path.join(anno_base_dir, args.classes)
    args.val_annotations = os.path.join(anno_base_dir, args.val_annotations)



    # args.annotations = os.path.join(anno_base_dir, 'train_prm.csv')
    # args.classes = os.path.join(anno_base_dir, 'mapping.csv')
    # args.val_annotations = os.path.join(anno_base_dir, 'validation_prm.csv')

    print('training csv {}, validation csv {}'.format(args.annotations, args.val_annotations))

    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
     # = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        # model.compile(
        #     loss={
        #         'regression': losses.smooth_l1(),
        #         'classification': losses.focal(),
        #         'mask': losses.mask_focal(),
        #     },
        #     optimizer=keras.optimizers.adam(lr=1e-4, clipnorm=0.001)  # used to be 1e-4
        # )
        training_model = model
        # training_model.compile(
        #     loss={
        #         'regression': losses.smooth_l1(),
        #         'classification': losses.focal(),
        #         'mask': losses.mask_focal(),
        #     },
        #     optimizer=keras.optimizers.adam(lr=1e-4, clipnorm=0.001)  # used to be 1e-4
        # )
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights

        # if weights is None and args.imagenet_weights:
        #     weights = backbone.download_imagenet()  # default to imagenet if nothing else is specified

        print('Creating model, this may take a second...')
        # model, training_model, prediction_model = create_models(
        #     args,
        #     backbone_retinanet=backbone.retinanet,
        #     num_classes=train_generator.num_classes(),
        #     weights=weights,
        #     multi_gpu=args.multi_gpu,
        #     freeze_backbone=args.freeze_backbone,
        #     config=args.config,
        #     model_config=model_config
        # )


    # print model summary
    # print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    # if 'vgg' in args.backbone or 'densenet' in args.backbone:
    #     train_generator.compute_shapes = make_shapes_callback(model)
    #     if validation_generator:
    #         validation_generator.compute_shapes = train_generator.compute_shapes

    # # create the callbacks
    # callbacks = create_callbacks(
    #     model,
    #     training_model,
    #     prediction_model,
    #     validation_generator,
    #     args,
    # )

    # print('outputs in prediction_model')
    # for o in prediction_model.outputs:
    #     print(o.shape)
    if args.snapshot:
        resume_epoch = int(args.snapshot.rsplit('/', -1)[-1].split('.')[0].rsplit('_', -1)[-1])

    elif args.weights:
        resume_epoch = int(args.weights.rsplit('/', -1)[-1].split('.')[0].rsplit('_', -1)[-1])
    else:
        resume_epoch = 0
    print('start from epoch:', resume_epoch)

    # conver generator to tf.data.dataset

    # dataset = tf.data.Dataset.from_generator( 
    #  generator=train_generator, output_types=(tf.float32,(tf.float32,tf.float32,tf.float32)), \
    #  output_shapes=(tf.TensorShape([2,512, 512, 3]), (tf.TensorShape([2,327360, 5]),tf.TensorShape([2,327360, 2]),tf.TensorShape([2,512, 512, 2]))))
    dataset = tf.data.Dataset.from_generator( 
     generator=train_generator, output_types=(tf.float32,(tf.float32,tf.float32)), \
     output_shapes=(tf.TensorShape([2,512, 512, 3]), (tf.TensorShape([2,327360, 5]), tf.TensorShape([2,512, 512, 2]))))

    # mapping function to format intput
    def preprocess(dataset):
        def element_fn(element1, element2):
            print (element2[0].shape)
            print (element2[1].shape)
            print (type(element2))
            return collections.OrderedDict([
            ('x', element1),
            ('y', element2),
            ])
            return np.array(element1), element2
        return dataset.repeat(3).apply(tf.contrib.data.unbatch()).map(element_fn).shuffle(100).batch(3)

    iterator = preprocess(dataset).make_one_shot_iterator()
    samples = iterator.get_next()

    # created for federated learning, please ignore it
    sample_batch = tf.contrib.framework.nest.map_structure(
        lambda x: x.numpy(), samples)

    # sess.run(tf.global_variables_initializer())
    # sample = sess.run(samples)
    # print (len(sample))
    # print (len(sample))
    # x = sample['x']
    # y = sample['y']
    # print(x.shape)
    # print(y[0].shape)
    # print(y[1].shape)
    # print(y[2].shape)

    def create_compiled_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(1, 3, padding = 'same',input_shape=(512, 512, 3))])
      
        def loss_fn(y_true, y_pred):
            return tf.keras.losses.MSE(y_true, y_pred)
 
     
        model.compile(
            loss=[losses.mask_focal_loss,losses.mask_focal_loss],
            optimizer=gradient_descent.SGD(learning_rate=0.02))
        return model

    init = tf.global_variables_initializer()

    def model_fn():
        training_model = create_models(
                        args,
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            config=args.config,
            model_config=model_config)
        # training_model = create_compiled_keras_model()
        # init = tf.global_variables_initializer()

        # sess =  tf.Session(config=config_proto,graph = g)

        # sess.run(init)
        return  tff.learning.from_compiled_keras_model(training_model, sample_batch)
 

    iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    state = iterative_process.initialize()

    for i in range(10):
        print(11)
        state, metrics = iterative_process.next(state, [dataset])
        print('round  {}, metrics={}'.format(str(i), metrics))


    #     emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    #     print (len(emnist_train.client_ids))
    #     print (emnist_train.output_types, emnist_train.output_shapes)

    #     example_dataset = emnist_train.create_tf_dataset_for_client(
    #         emnist_train.client_ids[0])


    #     NUM_EPOCHS = 10
    #     BATCH_SIZE = 20
    #     SHUFFLE_BUFFER = 500


    #     def preprocess(dataset):

    #       def element_fn(element):
    #         return collections.OrderedDict([
    #             ('x', tf.reshape(element['pixels'], [-1])),
    #             ('y', tf.reshape(element['label'], [1])),
    #         ])

    #       return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
    #           SHUFFLE_BUFFER).batch(BATCH_SIZE)
    #     nest = tf.contrib.framework.nest



    #     preprocessed_example_dataset = preprocess(example_dataset)
    #     print (11111111111111111111)

    #     print (preprocessed_example_dataset)
    #     print (11111111111111111111)

    #     iterator = preprocessed_example_dataset.make_one_shot_iterator()
    #     samples = iterator.get_next()

    #     sample_batch = nest.map_structure(
    #         lambda x: x, samples)

    #     sample_batch

    #     def make_federated_data(client_data, client_ids):
    #       return [preprocess(client_data.create_tf_dataset_for_client(x))
    #               for x in client_ids]

    #     NUM_CLIENTS = 3

    #     sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

    #     federated_train_data = make_federated_data(emnist_train, sample_clients)

    #     print (len(federated_train_data), federated_train_data[0])

    #     def create_compiled_keras_model():
    #       model = tf.keras.models.Sequential([
    #           tf.keras.layers.Dense(
    #               10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
          
    #       def loss_fn(y_true, y_pred):
    #         return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
    #             y_true, y_pred))
         
    #       model.compile(
    #           loss=loss_fn,
    #           optimizer=gradient_descent.SGD(learning_rate=0.02),
    #           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    #       return model
    #     init = tf.global_variables_initializer()
    # sess =  tf.Session(config=config_proto,graph = g)

    # def model_fn():
    #   keras_model = create_compiled_keras_model()
    #   sess.run(init)

    #   return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    # iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    # state = iterative_process.initialize()


    # state, metrics = iterative_process.next(state, federated_train_data)
    # print('round  1, metrics={}'.format(metrics))

    # print (next(dataset))
    # training_model.fit(
    #     x=x,
    #     y=y
    #     # steps_per_epoch=args.steps,
    #     # epochs=args.epochs,
    #     # verbose=1,
    #     # callbacks=callbacks,
    #     # initial_epoch=resume_epoch
    # )
    # print('Finish training')
    # return training_model, prediction_model


if __name__ == '__main__':
    main()
