import time
start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
  
#os.chdir('./object_detection')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_MODEL = '../../model.pb'
PATH_TO_LABELS = '../../gun.pbtxt'

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

PATH_TO_TEST_IMAGES_DIR = '../../test_images'
os.chdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGES = os.listdir('./')

IMAGE_SIZE = (12, 8)

output_image_path = ('../output_images')
output_csv_path = ('../output_csvs')

if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)
    
if not os.path.exists(output_csv_path):
    os.makedirs(output_csv_path)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        data = pd.DataFrame()
        for entry in TEST_IMAGES:
          image = Image.open(entry)
          width, height = image.size
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
              
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8,
              min_score_thresh=0.65)
          #write images
          #save the detection result images
          
          cv2.imwrite(output_image_path + '/' + os.path.basename(entry),image_np)
          
          s_boxes = boxes[scores > 0.9]
          s_classes = classes[scores > 0.9]
          s_scores = scores[scores > 0.9]
          
          #write table
          #Save the location coordinate results to the.csv table
          for i in range(len(s_classes)):

              newdata= pd.DataFrame(0, index=range(1), columns=range(7))
              newdata.iloc[0,0] = entry.split("\\")[-1].split('.')[0]
              newdata.iloc[0,1] = s_boxes[i][0]*height  #ymin
              newdata.iloc[0,2] = s_boxes[i][1]*width     #xmin
              newdata.iloc[0,3] = s_boxes[i][2]*height    #ymax
              newdata.iloc[0,4] = s_boxes[i][3]*width     #xmax
              newdata.iloc[0,5] = s_scores[i]
              newdata.iloc[0,6] = s_classes[i]
    
              data = data.append(newdata)
          data.to_csv(output_csv_path + '/' + os.path.basename(entry) + '.csv',index = False)
      
count = 0

for element in scores:
    if element[0] > 0.65:
        count = count + 1
    
end =  time.time()
print("Execution Time: ", end - start)
print(count)