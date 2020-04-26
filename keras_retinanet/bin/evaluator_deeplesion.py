import os
from collections import defaultdict

import argparse
import tensorflow.keras
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle

from matplotlib.markers import MarkerStyle
from skimage import measure
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
# print(os.path.dirname(__file__))
# sys.path.insert(0, '../')
# sys.path.append('../../')
# print(sys.path)
# print(os.path.join('./keras_retinanet.bin', '..', '..'))
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
#     sys.path.insert(0, os.path.join('./keras_retinanet.bin', '..', '..'))
#     from .. import deeplesion   # noin__" and __package__qa: F401
    # __package__ = "deeplesion"

# from visualiser import load_model_from_args, bb_intersection_over_union
from deeplesion.visualiser import load_model_from_args
from deeplesion.visualiser import bb_intersection_over_union
from deeplesion.generator import rreplace
from keras_retinanet.utils.image import read_image_bgr, preprocess_image


BASE_PATH = '/research/dept6/yhlong/mrJiang/Improveing_RetinaNet_Code/'

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

WARM_UP = 100


def compute_froc(predictions, number_of_files, number_of_lesions):
    true_positives = list(map(lambda prediction: prediction[0], filter(lambda prediction: prediction[1], predictions)))

    false_positives = list(map(lambda prediction: prediction[0], filter(lambda prediction: not prediction[1], predictions)))

    total_fps, total_tps = [], []

    all_probabilities = sorted(set(false_positives + true_positives))

    for threshold in tqdm(all_probabilities[1:]):
        total_fps.append((np.asarray(false_positives) >= threshold).sum())
        total_tps.append((np.asarray(true_positives) >= threshold).sum())

    total_fps.append(0)
    total_tps.append(0)
    total_fps = np.asarray(total_fps) / number_of_files
    total_sensitivity = np.asarray(total_tps) / number_of_lesions

    return total_fps, total_sensitivity


def plot_froc(lines, filename):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Average Number of False Positives')
    ax.set_ylabel('Lesion detection sensitivity')
    ax.set_ylim(bottom=0.55, top=0.975)
    ax.set_xlim(right=16)

    legend = []

    for line in lines:
        if len(line['total_sensitivity']) < 10:
            ax.plot(line['total_fps'], line['total_sensitivity'], ls='--', linewidth=1, marker='D',
                    markerfacecolor='none', markeredgewidth=1, markersize=4, color=line['color'])
        else:
            ax.plot(line['total_fps'], line['total_sensitivity'], ls='-', linewidth=1, color=line['color'])

        if 'label' in line:
            legend.append(line['label'])

    ax.set_aspect(20)

    if legend:
        ax.legend(legend, fontsize='x-small')
    plt.savefig(filename)


def add_evaluator_args(parser):

    # parser.add_argument('--model', type=str, help='Path to the model to use.')
    # parser.add_argument('--source', type=str, help='Name of the data source folder.')
    # parser.add_argument('--config', type=str, help='Path to config file.')
    # parser.add_argument('--h5', help='Input is a h5 file.', action='store_true')

    parser.add_argument('--gpu', default='0', type=str, help='Specify which GPU to use.')
    parser.add_argument('--model', type=str, help='Path to the model to use.',
                        default=os.path.join(BASE_PATH, 'snapshots/vgg19_csv_14.h5'))
                        # default = os.path.join(BASE_PATH, 'results/snapshots_2mm_attention_unet_no_weights/vgg19_csv_30.h5'))
    parser.add_argument('--source', type=str, help='Name of the data source folder.',
                        default='/research/dept6/yhlong/mrJiang/DeepLesionDataSet/formatted_2mm')
    parser.add_argument('--config', type=str, help='Path to config file.',
                        default=os.path.join(BASE_PATH, 'code/deeplesion/config/anchors4.ini'))
    parser.add_argument('--h5', help='Input is a h5 file.', action='store_true')


def evaluate(args=None, model=None):
    parser = argparse.ArgumentParser(description='Generate images for training.')
    add_evaluator_args(parser)

    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # use this environment flag to change which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set the modified tf session as backend in keras
    # keras.backend.tensorflow_backend.set_session(get_session())

    # df = pd.DataFrame.from_csv('DL_info.csv')
    df = pd.read_csv(os.path.join(BASE_PATH, 'code/deeplesion/DL_info.csv'))

    if not model:
        model = load_model_from_args(args)

    total_lesions = 0
    to_predict = defaultdict(list)
    total_sizes = defaultdict(int)
    total_types = defaultdict(int)
    predictions = []

    for row in tqdm(df.iterrows()):
        if row[1]['Train_Val_Test'] == 3 and row[1]['Possibly_noisy'] == 0:
            total_lesions += 1

            # path = '{}/{}'.format(args.source, rreplace(row[0], '_', '/', 1))
            path = '{}/{}'.format(args.source, rreplace(row[1]['File_name'], '_', '/', 1))

            scaling = 512 / int(row[1]['Image_size'].split(',')[0])
            position = [max(min(int(round(float(entry) * scaling)), 512), 0) for entry in row[1]['Bounding_boxes'].split(',')]

            current_scale = float(row[1]['Spacing_mm_px_'].split(',')[0])
            diameter = [float(entry) * current_scale for entry in row[1]['Lesion_diameters_Pixel_'].split(',')]
            min_size = min(diameter[0], diameter[1])
            lesion_type = row[1]['Coarse_lesion_type']

            if min_size < 10:
                total_sizes['lt10'] += 1
            elif min_size > 30:
                total_sizes['gt30'] += 1
            else:
                total_sizes['other'] += 1

            total_types[lesion_type] += 1

            to_predict[path].append({'position': position, 'min_size': min_size, 'lesion_type': lesion_type})

    inference_times = []
    index = -1

    positive_skipped = 0
    negative_skipped = 0
    positive_overlap = [[], []]
    negative_overlap = [[], []]

    for path in tqdm(to_predict):
        index += 1
        entry = to_predict[path]

        if args.h5:
            path = path.replace('png', 'h5')

        if os.path.isfile(path):
            original = read_image_bgr(path)

            # preprocess image for network
            image = preprocess_image(original, mode='tf')

            # process image
            if index > WARM_UP:
                start = time.clock()

            boxes, scores, labels, mask = model.predict_on_batch(np.expand_dims(image, axis=0))
            mask = mask[0]
            mask /= np.max(mask)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

            labeled = measure.label(mask)
            regions = measure.regionprops(labeled)

            if index > WARM_UP:
                inference_times.append(time.clock() - start)

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if label == -1:
                    break

                found = False

                best_region = 0
                for region in regions:
                    mask_box = [region.bbox[1], region.bbox[0], region.bbox[4], region.bbox[3]]
                    best_region = max(best_region, bb_intersection_over_union(mask_box, box))

                for i in range(len(entry)):
                    position = entry[i]['position']
                    if bb_intersection_over_union(position, box) >= 0.5:
                        predictions.append((score * (1 + best_region), True, entry[i]['min_size'], entry[i]['lesion_type'], best_region))
                        found = True
                        del entry[i]
                        break

                if found and best_region == 0:
                    positive_skipped += 1
                elif not found and best_region == 0:
                    negative_skipped += 1
                elif found:
                    positive_overlap[0].append(score)
                    positive_overlap[1].append(best_region)
                else:
                    negative_overlap[0].append(score)
                    negative_overlap[1].append(best_region)

                if not found:
                    predictions.append((score * (1 + best_region), False, None, None, best_region))
        else:
            print('WARNING: Missing file.')

    print('Positive skipped: ', positive_skipped)
    print('Negative skipped: ', negative_skipped)

    predictions = sorted(predictions, reverse=True)
    total_files = len(to_predict.keys())

    print('Total files: ', total_files)
    print('Total lesions: ', total_lesions)
    print('Inference time (ms): ', np.average(inference_times) * 1000)

    for fp_rate in [0.5, 1, 2, 4, 5, 8, 16]:
        possible_fps = int(round(fp_rate * total_files))
        correct = 0
        correct_sizes = defaultdict(int)
        correct_types = defaultdict(int)
        threshold = -1

        for i in range(len(predictions)):
            if predictions[i][1]:
                correct += 1

                if predictions[i][2] < 10:
                    correct_sizes['lt10'] += 1
                elif predictions[i][2] > 30:
                    correct_sizes['gt30'] += 1
                else:
                    correct_sizes['other'] += 1

                correct_types[predictions[i][3]] += 1
            else:
                possible_fps -= 1

            if possible_fps == 0:
                threshold = predictions[i][0]
                break

        for key in correct_sizes.keys():
            correct_sizes[key] /= total_sizes[key]

        for key in correct_types.keys():
            correct_types[key] /= total_types[key]

        print()
        print('{} FPs Sensitivity: {} Score threshold: {} '.format(fp_rate, correct / total_lesions, threshold))
        print('Sensitivity per size: <10: {}, 10-30: {} >30: {}'.format(
            correct_sizes['lt10'], correct_sizes['other'], correct_sizes['gt30']))
        print(correct_types)

    predictions = list(filter(lambda x: x[0] >= threshold,  predictions))
    total_fps, total_sensitivity = \
        compute_froc(predictions, number_of_files=total_files, number_of_lesions=total_lesions)

    data = {'total_fps': total_fps, 'total_sensitivity': total_sensitivity, 'color': 'blue'}

    with open(os.path.join(BASE_PATH,'code/deeplesion','froc/{}'.format(args.model.split('/')[-2]+'_'+args.model.split('/')[-1].replace('.h5', '.pickle'))), 'wb') as handle:
        pickle.dump(data, handle)

    plot_froc([data], os.path.join(BASE_PATH,'code/deeplesion','froc/{}'.format(args.model.split('/')[-2]+'_'+args.model.split('/')[-1].replace('.h5', '.png'))))


# if __name__ == "__main__":
# evaluate()

