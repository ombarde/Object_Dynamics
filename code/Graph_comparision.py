import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_spatio_temporal_features(frames):
    features = []
    for i in range(len(frames) - 1):
        frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 5, 3, 5, 1.2, 0)

        # Further reduce downsampling size
        flow_downsampled = cv2.resize(flow, (5, 5), interpolation=cv2.INTER_AREA)
        features.append(flow_downsampled)

    return np.array(features, dtype=np.float32)

# Load video from URL (you can replace it with your own video file)
video_path = 'moving_man.mp4'
cap = cv2.VideoCapture(video_path)

# Parameters
num_frames = 50
batch_size = 5

# Process video in batches
num_batches = num_frames // batch_size

# Extract spatio-temporal features
all_features = []
for batch in range(num_batches):
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Exiting...")
            break
        frames.append(cv2.resize(frame, (640, 480)))

    # Extract spatio-temporal features
    features = extract_spatio_temporal_features(frames)
    all_features.extend(features)

# Convert to a 2D array
all_features_2d = np.reshape(all_features, (len(all_features), -1))

# Scale the features
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features_2d)

# Reduce dimensionality using PCA (for visualization)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_features_scaled)

# GMM clustering
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(all_features_scaled)

# Mean Shift clustering
ms = MeanShift()
ms_labels = ms.fit_predict(all_features_scaled)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(all_features_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(all_features_scaled)

# Release video capture
cap.release()

# Visualize clustering results using PCA
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot GMM clustering results
plt.subplot(2, 2, 1)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=gmm_labels, cmap='viridis', s=30)
plt.title('GMM Clustering')

# Plot Mean Shift clustering results
plt.subplot(2, 2, 2)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=ms_labels, cmap='viridis', s=30)
plt.title('Mean Shift Clustering')

# Plot DBSCAN clustering results
plt.subplot(2, 2, 3)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=dbscan_labels, cmap='viridis', s=30)
plt.title('DBSCAN Clustering')

# Plot KMeans clustering results
plt.subplot(2, 2, 4)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=kmeans_labels, cmap='viridis', s=30)
plt.title('KMeans Clustering')

plt.tight_layout()
plt.show()