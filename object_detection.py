#This code is adapted from Priyanka Dwidedi
#https://github.com/priya-dwivedi/Deep-Learning/blob/master/Object_Detection_Tensorflow_API.ipynb

import os
import inspect
import cv2
#import time
#import argparse
#import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PIL import Image

import download_and_extract as dae
import frame_extract as fe

def model_name(model):
	if model == "mobilenet":
		return dae.URL_mobilenet.split('/')[-1]
	elif model == "inception":
		return dae.URL_inception.split('/')[-1]
	elif model == "rfcn_resnet":
		return dae.URL_rfcn_resnet.split('/')[-1]
	elif model == "rcnn_resnet":
		return dae.URL_rcnn_resnet.split('/')[-1]
	elif model == "rcnn_inception":
		return dae.URL_rcnn_inception.split('/')[-1]
	else:
		print "Error: Unexpected Model"
		sys.exit()

CURR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
INPUT_DIR = os.path.join(CURR_PATH,"input")
OUTPUT_DIR = os.path.join(CURR_PATH,"output")

# "mobilenet", "inception", "rfcn_resnet", "rcnn_resnet", "rcnn_inception" #
MODEL = "mobilenet"

dae.download_and_extract(MODEL)
fe.frame_extract()

### maybe need $protoc ... os.system(...)

MODEL_NAME = model_name(MODEL)
PATH_TO_CKPT = os.path.join(CURR_PATH, data, MODEL_NAME, "frozen_inference_graph.pb") #need function for right graph file
PATH_TO_LABELS = os.path.join(PATH, MODEL_NAME, "graph.pbtxt") #need function for right label file

NUM_CLASSES = 90 #maybe function?

label_map = label_map_util.load_labelmap("/home/sonat/tensorflow/models/object_detection/data/mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def detect_objects(image_np, sess, detection_graph):
	image_np_expanded = np.expand_dims(image_np, axis=0)
	image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

	boxes = detection_graph.get_tensor_by_name("detection_boxes:0")

	scores = detection_graph.get_tensor_by_name("detection_scores:0")
	classes = detection_graph.get_tensor_by_name("detection_classes:0")
	num_detections = detection_graph.get_tensor_by_name("num_detections:0")

	(boxes, scores, classes, num_detections) = sess.run(
		[boxes, scores, classes, num_detections],
		feed_dict={image_tensor:image_np_expanded})

	vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8)

	#Writing to a file with the desired annotation format

	im_width = image_np.shape[1]
	im_height = image_np.shape[0]
	sqBoxes = np.squeeze(boxes)
	sqScores = np.squeeze(scores)
	file = open("/home/sonat/Desktop/annotations.txt","a")
	for i in range(0,(sqBoxes).shape[0]):
		if sqScores[i] > 0.5:   #min_scores_thresh
			sqClasses = np.squeeze(classes).astype(np.int32)
			if sqClasses.all() in category_index.keys():
				file.write(str(int(sqBoxes[i][1]*im_width)) + ' ' + str(int(sqBoxes[i][0]*im_height)) +' ') 
				file.write(str(int(sqBoxes[i][3]*im_width)) + ' ' + str(int(sqBoxes[i][2]*im_height)) + ' ')
				file.write('_ _ _ _ ')
				file.write('\"' + category_index[sqClasses[i]]['name'] + '\"\n')
	return image_np


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = os.path.join(PATH,"/home/sonat/Desktop/Object_Detection/Test_Images")
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, "img_{}.jpg".format(i)) for i in range(1,4)]

IMAGE_SIZE = (12, 8)

for image_path in TEST_IMAGE_PATHS:
	image = Image.open(image_path)
	image_np = load_image_into_numpy_array(image)

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def,name='')

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		for image_path in TEST_IMAGE_PATHS:
			image = Image.open(image_path)
			image_np = load_image_into_numpy_array(image)
			image_process = detect_objects(image_np, sess, detection_graph)

			plt.figure(figsize=IMAGE_SIZE)
			cv2.imwrite("lol_"+image_path,image_np)git pul
