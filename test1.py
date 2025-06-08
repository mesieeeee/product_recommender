import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
import os
import cv2
from collections import defaultdict

# ---------------------------
# Step 1: Load embeddings and filenames
# ---------------------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ---------------------------
# Step 2: Build the KMeans model
# ---------------------------
num_clusters = 35  # choose based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(feature_list)
centroids = kmeans.cluster_centers_

# ---------------------------
# Step 3: Build Inverted Index
# ---------------------------
inverted_index = defaultdict(list)
for idx, cluster_id in enumerate(cluster_assignments):
    inverted_index[cluster_id].append(idx)

# ---------------------------
# Step 4: Define model for feature extraction
# ---------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ---------------------------
# Step 5: Function to extract normalized embedding from image path
# ---------------------------
def extract_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# ---------------------------
# Step 6: IVF Search Function
# ---------------------------
def ivf_search(query_embedding, centroids, inverted_index, feature_list, top_k_clusters=3, top_n_results=5):
    # Find top-k closest clusters
    cluster_distances = np.linalg.norm(centroids - query_embedding, axis=1)
    closest_clusters = np.argsort(cluster_distances)[:top_k_clusters]

    # Gather all vectors in these clusters
    candidate_indices = []
    for cluster_id in closest_clusters:
        candidate_indices.extend(inverted_index[cluster_id])

    candidate_vectors = feature_list[candidate_indices]
    distances = np.linalg.norm(candidate_vectors - query_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:top_n_results]

    return [candidate_indices[i] for i in nearest_indices]

# ---------------------------
# Step 7: Run a Sample Query
# ---------------------------
query_image_path = 'D:/yolo-project/yolo-project/yolo-project/clutches1.webp'
query_embedding = extract_embedding(query_image_path)

matched_indices = ivf_search(
    query_embedding,
    centroids,
    inverted_index,
    feature_list,
    top_k_clusters=3,
    top_n_results=5
)

print("Top Matches:")
old_base_path = 'H:/archive/images'
new_base_path = 'D:/yolo-project/yolo-project/yolo-project/images'

for idx in matched_indices:
    old_path = filenames[idx]
    filename_only = os.path.basename(old_path)
    new_path = os.path.join(new_base_path, filename_only)

    temp_img = cv2.imread(new_path)
    if temp_img is None:
        print(f"❌ Failed to load: {new_path}")
        continue

    cv2.imshow('Match', cv2.resize(temp_img, (224, 224)))
    cv2.waitKey(0)

cv2.destroyAllWindows()

