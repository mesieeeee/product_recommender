import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('D:/yolo-project/yolo-project/yolo-project/chappals1.webp',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='cosine')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

import os

# Define the OLD base path (as saved inside filenames.pkl)
old_base_path = 'H:/archive/images'
# Define the NEW base path where the images are now
new_base_path = 'D:/yolo-project/yolo-project/yolo-project/images'

# When reading each filename, replace the old part with the new part
for file in indices[0][1:6]:
    old_path = filenames[file]
    filename_only = os.path.basename(old_path)  # just '30461.jpg'
    new_path = os.path.join(new_base_path, filename_only)

    temp_img = cv2.imread(new_path)
    if temp_img is None:
        print(f"❌ Failed to load: {new_path}")
        continue

    cv2.imshow('output', cv2.resize(temp_img, (224, 224)))
    cv2.waitKey(0)


for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(224, 224)))
    cv2.waitKey(0)
