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
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree as ET

class WeedConfig(Config):
    # Give the configuration a recognizable name
    NAME = "weed_detection" 
    
    NUM_CLASSES = 1 + 1 # background + classes

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = WeedConfig()

class WeedDetector(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class('dataset', 1, 'Weed')
    
        for i, filename in enumerate(os.listdir(dataset_dir)):
            if '.jpg' in filename:
                self.add_image('dataset', 
                image_id=i, 
                path=os.path.join(dataset_dir, filename), 
                annotation=os.path.join(dataset_dir, filename.replace('.jpg', '.xml')))
    
    def extract_boxes(self, filename):

        tree = ET.parse(filename)

        root = tree.getroot()

        boxes = []
        classes = []

        for member in root.findall('object'):
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(self.class_names.index(member[0].text))


        # extract image dimensions
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        return boxes, classes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        path = info['annotation']
        boxes, classes, w, h = self.extract_boxes(path)

        masks = np.zeros([h, w, len(boxes)], dtype='uint8')

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
        return masks, np.asarray(classes, dtype='int32')


    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# Create training and validation set
# train set
dataset_train = WeedDetector()  
dataset_train.load_dataset('C:\\Code\\Python\\Mask_RCNN\\weed\\images')
dataset_train.prepare()
print('Train: %d' % len(dataset_train.image_ids))
 
test_set = WeedDetector()
test_set.load_dataset("C:\\Code\\Python\\Mask_RCNN\\weed\\test", is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir="./")

#load the weights for COCO
model.load_weights("C:\\Users\\krish\\Downloads\\mask_rcnn_coco(1).h5",     
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, test_set, learning_rate=2*config.LEARNING_RATE, epochs=2, layers='heads')
history = model.keras_model.history.history

model_path = "C:\\Code\\Python\\Mask_RCNN"  + '.' + str(time.time()) + '.h5'
model.keras_model.save_weights(model_path)