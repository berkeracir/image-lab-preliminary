import os
import inspect
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PIL import Image

import sys
import tarfile

from six.moves import urllib

CURR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

INPUT_DIR = os.path.join(CURR_PATH,"test","videos")
ANNOTATION_DIR = os.path.join(CURR_PATH,"test","annotation")
FRAMES_DIR = os.path.join(CURR_PATH,"test","frames")
OUTPUT_DIR = os.path.join(CURR_PATH,"test","output")
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

URL_mobilenet = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz"
URL_inception = "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz"
URL_rfcn_resnet = "http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz"
URL_rcnn_resnet = "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz"
URL_rcnn_inception = "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz"

def frame_extract():
	path, dirs, files = os.walk(INPUT_DIR).next()
	for i in files:
		captured_video = cv2.VideoCapture(os.path.join(path, i))
		success, image = captured_video.read()
		count = 0
		directory = os.path.join(FRAMES_DIR)
		if not os.path.exists(directory):
			os.makedirs(directory)	
		while success:
			cv2.imwrite(directory + '/frame_%.4d.jpg' % count, image)
			success, image = captured_video.read()
			count += 1

def download_and_extract(model):
	if model == "mobilenet":
		URL = URL_mobilenet
	elif model == "inception":
		URL = URL_inception
	elif model == "rfcn_resnet":
		URL = URL_rfcn_resnet
	elif model == "rcnn_resnet":
		URL = URL_rcnn_resnet
	elif model == "rcnn_inception":
		URL = URL_rcnn_inception
	else:
		print "Error:Unexpected Model"
		sys.exit()
	dest_directory = os.path.join(CURR_PATH,"data")
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>Downloading: %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def model_name(model):
	if model == "mobilenet":
		return URL_mobilenet.split('/')[-1]
	elif model == "inception":
		return URL_inception.split('/')[-1]
	elif model == "rfcn_resnet":
		return URL_rfcn_resnet.split('/')[-1]
	elif model == "rcnn_resnet":
		return URL_rcnn_resnet.split('/')[-1]
	elif model == "rcnn_inception":
		return URL_rcnn_inception.split('/')[-1]
	else:
		print "Error: Unexpected Model"
		sys.exit()

# "mobilenet", "inception", "rfcn_resnet", "rcnn_resnet", "rcnn_inception" #
MODEL = "rcnn_inception"

download_and_extract(MODEL)
frame_extract()

MODEL_NAME = model_name(MODEL).split('.')[0]
PATH_TO_CKPT = os.path.join(CURR_PATH, "data", MODEL_NAME, "frozen_inference_graph.pb")
PATH_TO_LABELS = os.path.join(CURR_PATH, "data", MODEL_NAME, "graph.pbtxt")

NUM_CLASSES = 90

#label_map = label_map_util.load_labelmap("/home/vermithrax/Desktop/image-lab/data/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/graph.pbtxt")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#label_map = label_map_util.load_labelmap("/home/vermithrax/tensorflow/models/object_detection/data/mscoco_label_map.pbtxt") ### NOT RIGHT ###
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

	(im_height, im_width) = (image_np.shape[0], image_np.shape[1])
	sqBoxes = np.squeeze(boxes)
	sqScores = np.squeeze(scores)
	for i in range(0,(sqBoxes).shape[0]):
		if sqScores[i] > 0.5:   #min_scores_thresh
			sqClasses = np.squeeze(classes).astype(np.int32)
			if not os.path.exists(ANNOTATION_DIR):
				os.makedirs(ANNOTATION_DIR)
			file = open(os.path.join(ANNOTATION_DIR,category_index[sqClasses[i]]['name']+".txt"),"a")
			if sqClasses.all() in category_index.keys():
				file.write("x ")
				file.write(str(int(sqBoxes[i][1]*im_width)) + ' ' + str(int(sqBoxes[i][0]*im_height)) +' ') 
				file.write(str(int(sqBoxes[i][3]*im_width)) + ' ' + str(int(sqBoxes[i][2]*im_height)) + ' ')
				file.write(image_path.split('/')[-1].split('.')[0] + ' _ _ _ ')
				file.write('\"' + category_index[sqClasses[i]]['name'] + '\"\n')
			file.close()
	return image_np

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

tmp_path, dirs, TEST_IMAGE_PATHS = os.walk(FRAMES_DIR).next()
TEST_IMAGE_PATHS = sorted(TEST_IMAGE_PATHS)

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
			image = Image.open(os.path.join(tmp_path,image_path))
			image_np = load_image_into_numpy_array(image)
			image_process = detect_objects(image_np, sess, detection_graph)
			image_process = cv2.cvtColor(image_process, cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(OUTPUT_DIR, image_path),image_process)
