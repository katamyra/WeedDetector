from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import matplotlib
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree as ETx

#img = cv2.imread('C:\\Code\\Python\\Mask_RCNN\\weed\\test\\00007.jpg',0)
class WeedConfig(Config):
    # Give the configuration a recognizable name
    NAME = "weed_detection"
    
    NUM_CLASSES = 1 + 1 # background + classes

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = WeedConfig()


model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

model.load_weights('C:\\Code\\Python\\Mask_RCNN.1609520996.5985641.h5', by_name=True)
img = load_img('C:\\Code\\Python\\Mask_RCNN\\weed\\test\\00007.jpg')
img = np.asanyarray(img)
results = model.detect([img])

print(results)

