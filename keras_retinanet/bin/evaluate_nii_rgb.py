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
import os
import sys

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import SimpleITK as sitk

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_v2_behavior()


if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..utils.anchors import compute_overlap
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.eval import _compute_ap, _get_annotations
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.visualization import draw_detections, draw_annotations
from keras_retinanet.bin.train_edit import create_models
from keras_retinanet.utils.image import preprocess_image, resize_image

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def format_image(scalar, multiplier, image):
    h = image.shape[1]
    w = image.shape[0]

    for y in range(0, h):
        for x in range(0, w):
            image[x, y] = int(min(max((image[x, y] + scalar) / multiplier, 0), 255))

    return image


def create_generator(args):
    """ Create generators for evaluation.
    """

    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def _get_detections(args, generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    detection_out = np.zeros([generator.size(),512,512,3])
    # detection_out = np.zeros([generator.size(),512,512])
    attention_out = np.zeros([generator.size(),512,512])
    mask_out = np.zeros([generator.size(),512,512])

    for i in tqdm(range(generator.size()), desc='Running network: '):
        raw_image    = generator.load_image(i)
        # image = np.expand_dims(raw_image.copy(), axis=-1)
        # image = np.repeat(image, 3, axis=-1)
        # image        = generator.preprocess_image(image)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        boxes, scores, labels, masks, attention_map = model.predict_on_batch(np.expand_dims(image, axis=0))
        # print('scores:', scores.shape)
        # print('labels',labels.shape)

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]
        # print('indices', indices)
        scores = scores.numpy()
        boxes = boxes.numpy()
        labels = labels.numpy()
        masks = masks.numpy()
        attention_map = attention_map.numpy()
        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]
        # print(scores_sort)

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, score_threshold=args.detection_threshold, label_to_name=generator.label_to_name)


            detection_out[i, :, :] = raw_image

            attention_map[np.where(attention_map < args.attention_threshold)] = 0
            # attention_out[i, :, :] = cv2.flip( cv2.resize(np.squeeze(np.uint8(attention_map * 255)), (origin_shape[1], origin_shape[0])), 0)
            attention_out[i, :, :] = cv2.resize(np.squeeze(np.uint8(attention_map * 255)), (512, 512))

            masks[masks < args.segmentation_threshold] = 0
            masks = cv2.resize(np.squeeze(np.uint8(masks * 255)), (512, 512))

            mask_out[i, :, :] = masks

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
    if save_path is not None:
        detection_out = sitk.GetImageFromArray(detection_out)
        sitk.WriteImage(detection_out, os.path.join(save_path, 'detection_result.nii.gz'))

        attention_out = sitk.GetImageFromArray(attention_out)
        sitk.WriteImage(attention_out, os.path.join(save_path, 'attention_result.nii.gz'))

        mask_out = sitk.GetImageFromArray(mask_out)
        sitk.WriteImage(mask_out, os.path.join(save_path, 'masks_result.nii.gz'))

    return all_detections


def evaluate(
    args,
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(args, generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    # print('all detections:', all_detections)
    # print('all all_annotations:', all_annotations)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    # print('###########')
    # print(generator.num_classes())

    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            print('generator has not label')
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        false_negatives = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    # print('NO bbox annos')
                    # print(d)
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 0)
                    false_negatives = np.append(false_negatives, 1)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # print(num_annotations)
        # print(all_detections[0][label].shape)
        # print(all_detections[0][label])
        TP = sum(true_positives)
        FP = sum(false_positives)
        FN = sum(false_negatives)
        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)


        print('TP:', TP)
        print('FP:', FP)
        print('FN:', FN)
        print('Recall:', TP / (TP + FN))
        print('Precision:', TP / (TP + FP))

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions



def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('--model', help='Path to RetinaNet model.', default=None)
    parser.add_argument('--weights', help='only load weights.', default=None)
    parser.add_argument('--nii',              help='path to nii files.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='vgg19')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--detection-threshold', help='Threshold used for determining what detections to draw.', default=0.4, type=int)
    parser.add_argument('--segmentation-threshold', help='Threshold used for filter segmentation map.', default=0.1, type=int)
    parser.add_argument('--attention-threshold', help='Threshold used for filter attention map.', default=0.8, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).',default=None)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=512)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=512)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    return parser.parse_args(args)


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
    # keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # anno_base_dir = '/Users/jemary/Data/DataSet/COVID_Data/'
    anno_base_dir = '/research/dept8/qdou/mrjiang/COVIDDataSet/'

    args.annotations = os.path.join(anno_base_dir, args.annotations)
    args.classes = os.path.join(anno_base_dir, args.classes)


    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

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
    if args.dataset_type == 'csv':
        average_precisions = evaluate(
            args,
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path,
        )

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        if args.weighted_average:
            print('mAP: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        else:
            print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

if __name__ == '__main__':
    main()
