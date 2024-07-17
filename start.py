from __future__ import print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
import numpy as np
import pandas as pd 
import nibabel
import random
import sys

import keras
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, concatenate, Conv2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from keras import backend as K
from keras.callbacks import History

import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split



import skimage
from skimage import img_as_ubyte, img_as_float32, img_as_uint
import skimage.transform 
from skimage.transform import resize
import skimage.io 
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage import io
from scipy import ndimage as ndi
from skimage.morphology import label, ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from skimage.io import imread
import scipy.misc

import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from glob import glob

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Scikit-learn :', sklearn.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)