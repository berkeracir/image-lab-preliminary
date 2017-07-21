import cv2
import os

path, dirs, files = os.walk('/home/sonat/Desktop/input/').next()
for i in files:
	captured_video = cv2.VideoCapture(path+i)
	success,image = captured_video.read()
	count = 0
	success = True
	directory = '/home/sonat/Desktop/output/'+ i
	if not os.path.exists(directory):
		os.makedirs(directory)	
	while success:
		success,image = captured_video.read()
		cv2.imwrite(directory + '/frame%d.jpg' % count, image)
		count += 1