import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.misc import pilutil
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
import tensorflow as tf
import skimage.feature
import cv2
import datetime
from tensorflow.contrib.learn.python.learn.datasets import mnist

num_examples = 1

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, name):
    rows = images.shape[0]
    cols = images.shape[1]
    depth = images.shape[2]

    filename = os.path.join(name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    image_raw = images.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image_and_labels_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    writer.close()


image_files = os.listdir(".")
for picture_number in image_files:
    try:
        print(datetime.datetime.now())
        if "npy" in picture_number:
            image_data = np.load(picture_number)
            convert_to(image_data, picture_number)
    except IOError as e:
        print('Could not read:', picture_number, ':', e, '- it\'s ok, skipping.')

