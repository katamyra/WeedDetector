# evaluate the mask rcnn model on the kangaroo dataset
import time
start1 = time.time()
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import skimage
import numpy as np
import pdb
import cv2

import glob
import pdb
import os
import re

# class that defines and loads the weed dataset
class WeedDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define classes
		self.add_class("dataset", 1, "crop")
		self.add_class("dataset", 2, "weed")
		# define data locations
		images_dir = 'C:\\Code\\Python\\Mask_RCNN\\weed\\images'
		annotations_dir = 'C:\\Code\\Python\\Mask_RCNN\\weed\\annots'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if (image_id == '.ipynb_checkpo'):
				continue
			# skip all images after 115 if we are building the train set
			if is_train and int(image_id) >= 115:
				continue
			# skip all images before 115 if we are building the test/val set
			if not is_train and int(image_id) < 115:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		#for box in root.findall('.//bndbox'):
		for box in root.findall('.//object'):
			name = box.find('name').text
			xmin = int(box.find('./bndbox/xmin').text)
			ymin = int(box.find('./bndbox/ymin').text)
			xmax = int(box.find('./bndbox/xmax').text)
			ymax = int(box.find('./bndbox/ymax').text)
			#coors = [xmin, ymin, xmax, ymax, name]
			coors = [xmin, ymin, xmax, ymax, name]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		#pdb.set_trace()
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			if (box[4] == 'crop'):
				masks[row_s:row_e, col_s:col_e, i] = 2
				class_ids.append(self.class_names.index('crop'))
			else:
				masks[row_s:row_e, col_s:col_e, i] = 1
				class_ids.append(self.class_names.index('weed'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "weed_cfg"
	# number of classes (background + weed + crop)
	NUM_CLASSES = 1 + 2
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=1):
	times = []
	imgExt = 'png'
	input_path = 'C:\\Code\\Python\\Mask_RCNN\\imgs'
	output_path = "C:\\Code\\Python\\Mask_RCNN\\imgout"
	nameR = input_path #.replace('*.'+imgExt,'')
	pathR, dirsR, filesR = next(os.walk(nameR))
	file_count = len(filesR)
	#pdb.set_trace()

	img_num = 0     #file number initial
	
	cnt = 0
	del_img = []
	file_name = '%05d.'+imgExt
	output_path = output_path+file_name     #Output Image Filename
	#pdb.set_trace()
	sort_temp = []
	def natural_sort(l): 
		convert = lambda text: int(text) if text.isdigit() else text.lower() 
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		return sorted(l, key = alphanum_key)

	for img_jpg1 in glob.glob(input_path):
			sort_temp.append(img_jpg1)
	sort = natural_sort(sort_temp)

	for img_jpg in sort:

			filename = output_path%int(img_num)        #Output Image filename
			cnt+=1
			img_num+=1
			print ('Renamed: ',cnt,'Out of: ',file_count); print("a:", img_jpg)

			image = skimage.io.imread(img_jpg) #dataset.load_image(i+15)
			overlay = image.copy()
			sample = expand_dims(image, 0)
			start3 = time.time()
			yhat = model.detect(sample, verbose=0)[0]
			delta = (time.time() - start3)
			times.append(delta)
			boxCount = 0
			alpha = 0.4
			for classId in yhat['class_ids']:
				if (classId == 2):
					box = yhat['rois'][boxCount]
					# get coordinates
					y1, x1, y2, x2 = box
					cv2.rectangle(overlay, (x1,y1), (x2,y2), (150,0,0) , -1)
					predictedImage = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
					predictedImage = cv2.cvtColor(predictedImage,cv2.COLOR_BGR2RGB)
				else:
					box = yhat['rois'][boxCount]
					# get coordinates
					y1, x1, y2, x2 = box
					cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,150) , -1)
					predictedImage = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
					predictedImage = cv2.cvtColor(predictedImage,cv2.COLOR_BGR2RGB)
				cv2.imwrite(filename,predictedImage)
				boxCount+=1

# load the train dataset
train_set = WeedDataset()
train_set.load_dataset('weed', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = WeedDataset()
test_set.load_dataset('weed', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = "C:\\Users\krish\\Downloads\\mask_rcnn_trained_weed_model.h5"
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
