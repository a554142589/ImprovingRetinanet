#!/usr/bin/env python

import argparse
import os
import sys

import keras
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2
import SimpleITK as sitk
import csv
# Allow relative imports when being executed as script.


if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.visualization import draw_box, label_color, draw_caption
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.bin.train_edit import create_models


def format_image(scalar, multiplier, image):
    h = image.shape[1]
    w = image.shape[0]

    for y in range(0, h):
        for x in range(0, w):
            image[x, y] = int(min(max((image[x, y] + scalar) / multiplier, 0), 255))

    return image


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, slice_id=None, bbox_writer=None, score_threshold=0.4):  # score_threshold used to be 0.5
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]
    for i in selection:
        c = color if color is not None else label_color(labels[i])

        if bbox_writer is not None and slice_id is not None:
            tar_path = 'slice_{}.png'.format(slice_id)
            b = np.array(boxes[i, :]).astype(int)
            bbox_writer.writerow([tar_path]+ [b[0],b[1],b[2],b[3]]+['lesion'])

        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else str(labels[i])) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


# def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, config=None, model_config=None):
#     """ Creates three models (model, training_model, prediction_model).
#     Args
#         backbone_retinanet : A function to call to create a retinanet model with a given backbone.
#         num_classes        : The number of classes to train.
#         weights            : The weights to load into the model.
#         multi_gpu          : The number of GPUs to use for training.
#         freeze_backbone    : If True, disables learning for the backbone.
#         config             : Config parameters, None indicates the default configuration.
#
#     Returns
#         model            : The base model. This is also the model that is saved in snapshots.
#         training_model   : The training model. If multi_gpu=0, this is identical to model.
#         prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
#     """
#
#     if model_config is None:
#         model_config = dict()
#
#     modifier = None
#
#     # load anchor parameters, or pass None (so that defaults will be used)
#     anchor_params = None
#     num_anchors = None
#     if config and 'anchor_parameters' in config:
#         anchor_params = parse_anchor_parameters(config)
#         num_anchors = anchor_params.num_anchors()
#
#     # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
#     # optionally wrap in a parallel model
#     channels = None
#     if config and 'dimensions' in config:
#         channels = int(config['dimensions']['channels'])
#
#     if multi_gpu > 1:
#         from keras.utils import multi_gpu_model
#         with tf.device('/cpu:0'):
#             model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, channels=channels, model_config=model_config),
#                                        weights=weights, skip_mismatch=True)
#         # training_model = multi_gpu_model(model, gpus=multi_gpu)
#     else:
#         model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, channels=channels, model_config=model_config), weights=weights, skip_mismatch=True)
#         # training_model = model
#
#
#     return model


# def create_generator(args):
#     """ Create generators for evaluation.
#     """
#     if args.dataset_type == 'csv':
#         validation_generator = CSVGenerator(
#             args.annotations,
#             args.classes,
#             image_min_side=args.image_min_side,
#             image_max_side=args.image_max_side,
#             config=args.config,
#             shuffle_groups=False
#         )
#     elif args.dataset_type == 'nii':
#         return None
#     else:
#         raise ValueError('Invalid data type received: {}'.format(args.dataset_type))
#
#     return validation_generator
#

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    nii_parser = subparsers.add_parser('nii')
    nii_parser.add_argument('source', help='Path to nii files for evaluation.')

    parser.add_argument('--model',               help='Path to RetinaNet model.', default=None)
    parser.add_argument('--weights',             help='only load weights.', default=None)
    parser.add_argument('--convert-model',       help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',            help='The backbone of the model.', default='vgg19')
    parser.add_argument('--gpu',                 help='Id of the GPU to use (as reported by nvidia-smi).', default='0')
    parser.add_argument('--multi-gpu',           help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--detection-threshold', help='Threshold used for determining what detections to draw.', default=0.4, type=int)
    parser.add_argument('--attention-threshold', help='Threshold used for filter attention map.', default=0.8, type=int)
    parser.add_argument('--save-path',           help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',      help='Rescale the image so the smallest side is min_side.', type=int, default=512)
    parser.add_argument('--image-max-side',      help='Rescale the image if the largest side is larger than max_side.', type=int, default=512)
    parser.add_argument('--config',              help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--weighted-average',    help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    return parser.parse_args(args)


def _get_detections(args, model, score_threshold=0.05, max_detections=100, filename=None, save_path=None):
    """ Get the detections and attentions map
    # Arguments
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Outputs:
        detection and attention results in nii.gz format
    """

    sitk_img = sitk.ReadImage(filename)
    volume = sitk.GetArrayFromImage(sitk_img)
    slice_range = volume.shape[0]
    print('Volumes has {} slicecs in total'.format(slice_range))
    origin_shape = volume.shape[1:]

    window = [-1050, 1024]
    multiplier = (window[1] - window[0]) / 255  # （1050 - -1024） /255
    scalar = -window[0]  # 1024

    detection_out = np.zeros(volume.shape)
    attention_out = np.zeros(volume.shape)
    mask_out = np.zeros(volume.shape)

    results = open(os.path.join(save_path, 'output_bbox.csv'), 'w', newline='')
    result_writer = csv.writer(results, delimiter=',')

    for i in tqdm(range(slice_range), desc='Running network:'):
    # for i in range(125,130):
        raw_image    = volume[i,:,:]

        image_cut = np.int64(raw_image)
        image_cut = format_image(scalar, multiplier, image_cut)
        image_cut = np.uint8(image_cut)

        # change 1 channel to 3
        image = np.expand_dims(image_cut,axis=-1)
        image = np.repeat(image, 3, axis=-1)

        image = preprocess_image(image.copy())
        image, scale = resize_image(image,min_side=args.image_min_side,max_side=args.image_max_side)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        boxes, scores, labels, masks, attention_map = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]
        # print('indices', indices)

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]

        draw_detections(raw_image, image_boxes, image_scores, image_labels, slice_id=i, bbox_writer=result_writer, score_threshold=args.detection_threshold)

        detection_out[i,:,:] = cv2.flip(raw_image,0)

        # attention_map[np.where(attention_map < args.attention_threshold)] = 0
        # attention_out[i,:,:] = cv2.flip(cv2.resize(np.squeeze(np.uint8(attention_map * 255)), (origin_shape[1],origin_shape[0])), 0)
        #
        # mask_out[i,:,:] = cv2.flip(cv2.resize(np.squeeze(np.uint8(masks * 255)), (origin_shape[1],origin_shape[0])), 0)
    results.flush()
    results.close()
    detection_out = sitk.GetImageFromArray(detection_out)
    detection_out.CopyInformation(sitk_img)
    sitk.WriteImage(detection_out, os.path.join(save_path, 'detection_'+str(filename.rsplit('/',-1)[-1])))

    # attention_out = sitk.GetImageFromArray(attention_out)
    # attention_out.CopyInformation(sitk_img)
    # sitk.WriteImage(attention_out,os.path.join(save_path, 'attention_'+str(filename.rsplit('/',-1)[-1])))

    # mask_out = sitk.GetImageFromArray(mask_out)
    # mask_out.CopyInformation(sitk_img)
    # sitk.WriteImage(mask_out, os.path.join(save_path, 'masks_' + str(filename.rsplit('/', -1)[-1])))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    if args.model is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.model, backbone_name=args.backbone)
    elif args.weights is not None:
        weights = args.weights

        print('Creating model and Loading weights, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            # note : when mapping.csv only contains lesion,0,  generator.num_classes() ==1
            num_classes=1,
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=False,
            config=args.config,
            model_config={}
        )
    else:
        raise ValueError("You have to specify a model")

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    # print model summary
    # print(model.summary())

    # start evaluation
    if args.dataset_type == 'nii':
        _get_detections(args, model, filename=args.source, save_path=args.save_path)
        # pass
    else:
        raise ValueError("Not supported dataset_type")


if __name__ == '__main__':
    main()
