import time
start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_MODEL = 'model_a.pb'
PATH_TO_LABELS = 'gun.pbtxt'

NUM_CLASSES = 6
      
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')    
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)

OPENPOSE_INPUT = ('/content/cs4243_miniproject_team16/openpose_input')
NORMAL_OUTPUT = ('/content/cs4243_miniproject_team16/normal')

if not os.path.exists(OPENPOSE_INPUT):
    os.makedirs(OPENPOSE_INPUT)
    
if not os.path.exists(NORMAL_OUTPUT):
    os.makedirs(NORMAL_OUTPUT)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
if not os.path.exists(PATH_TO_TEST_IMAGES_DIR):
    os.makedirs(PATH_TO_TEST_IMAGES_DIR)

os.chdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGES = os.listdir('./')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        data = pd.DataFrame()
        for entry in TEST_IMAGES:
          image = Image.open(entry)
          width, height = image.size
          image_np = load_image_into_numpy_array(image)
          image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          image_np_expanded = np.expand_dims(image_np, axis=0)
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
              
          if scores[0][0] >= 0.65:
            cv2.imwrite(OPENPOSE_INPUT + '/' + os.path.basename(entry), image_np)
          else:
            cv2.imwrite(NORMAL_OUTPUT + '/' + os.path.basename(entry), image_np)