import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

OPENPOSE_INPUT = '/content/cs4243_miniproject_team16/openpose_input/'

CARRYING_OUTPUT = '/content/cs4243_miniproject_team16/carrying/'
THREAT_OUTPUT = '/content/cs4243_miniproject_team16/threat/'

MODEL_B_INPUT = '/content/cs4243_miniproject_team16/model_b_input/'
MODEL_B_PATH = '/content/cs4243_miniproject_team16/model_b.h5'

mainList = []
for root, dirs, files in os.walk(MODEL_B_INPUT):
  for name in files:
    if name.endswith('.json'):
      with open(os.path.join(root, name), 'r') as f:
        data = json.load(f)

      for person in data['people']:
        mainList.append((root.split('/')[3], person['pose_keypoints_2d']))

x_test = [c[0] for c in mainList]
y_test = [e[1] for e in mainList]

one_hot_encoder = OneHotEncoder(sparse=False)
test_labels = one_hot_encoder.fit_transform(np.array(x_test).reshape(-1, 1))

input = tf.data.Dataset.from_tensor_slices((y_test, test_labels)).batch(1)

model_b = tf.keras.models.load_model(MODEL_B_PATH)
predictions = model_b.predict(input)
print(predictions)

if not os.path.exists(CARRYING_OUTPUT):
    os.makedirs(CARRYING_OUTPUT)

if not os.path.exists(THREAT_OUTPUT):
    os.makedirs(THREAT_OUTPUT)

entryNo = 0
for root, dirs, files in os.walk(MODEL_B_INPUT):
  for name in files:
    if name.endswith('.json'):
      file = os.path.basename(name).replace('_keypoints.json','.jpg')
      print(file)
      if predictions[entryNo, 1] >= 0.5:
        os.rename(OPENPOSE_INPUT + str(file), THREAT_OUTPUT + str(file))
      else:
        os.rename(OPENPOSE_INPUT + str(file), CARRYING_OUTPUT + str(file)) 
      entryNo = entryNo + 1;
