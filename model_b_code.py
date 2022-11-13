import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

mainList = []
for root, dirs, files in os.walk('/content/cs4243_miniproject_team16/model_b_input/'):
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

model_b = tf.keras.models.load_model('/content/cs4243_miniproject_team16/model_b.h5')
model_b.predict(input)