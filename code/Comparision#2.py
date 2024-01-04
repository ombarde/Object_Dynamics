import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

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

def cluster_frames(features, algorithm):
    if algorithm == 'GMM':
        model = GaussianMixture(n_components=2)
    elif algorithm == 'MeanShift':
        model = MeanShift()
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'KMeans':
        model = KMeans(n_clusters=2)

    features_flat = features.reshape(features.shape[0], -1)
    features_flat_normalized = StandardScaler().fit_transform(features_flat)

    labels = model.fit_predict(features_flat_normalized)

    return labels

def detect_objects_hog(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
    return found

# Load video from URL (you can replace it with your own video file)
video_path = 'moving_man.mp4'
cap = cv2.VideoCapture(video_path)

# Parameters
num_frames = 5  # Change this value to limit the number of frames
batch_size = 5
algorithms = ['GMM', 'MeanShift', 'DBSCAN', 'KMeans']
linear_model = LinearRegression()

# Process video in batches
num_batches = num_frames // batch_size

# Create a list to store concatenated frames for saving the final image
concatenated_frames_list = []

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

    # Check if features are not empty and there are enough frames for clustering
    if len(frames) > 1 and features.size > 0:
        # Train the Gaussian Mixture Model
        gmm = GaussianMixture(n_components=min(2, len(features) - 1))
        gmm.fit(features.reshape(features.shape[0], -1))

        # Train a linear regression model for further prediction
        X_train, y_train = features[:-1].reshape(-1, 5 * 5 * 2), features[1:].reshape(-1, 5 * 5 * 2)
        linear_model.fit(X_train, y_train)

    # Display the tracked object using GMM and linear regression predictions
    for i in range(len(frames)):
        if i == len(frames) - 1:
            break

        # Predict the next location using GMM
        gmm_prediction = gmm.predict(features[-1].reshape(1, -1))

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Detect people using HOG and SVM
        found = detect_objects_hog(gray_frame)

        # Draw rectangles around the detected objects
        for (x, y, w, h) in found:
            cv2.rectangle(frames[i], (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compare clustering results for each algorithm
        for j, algorithm in enumerate(algorithms):
            # Cluster frames
            labels = cluster_frames(features, algorithm)

            # Visualize the clustering results
            clustered_frames = []
            colors = np.random.randint(0, 255, size=(max(labels) + 1, 3))

            for label in labels:
                clustered_frames.append(frames[label])

            clustered_frames = np.concatenate(clustered_frames, axis=1)
            clustered_frames = cv2.putText(clustered_frames, f'{algorithm} Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(f'{algorithm} Clustering', clustered_frames)
            cv2.waitKey(500)

            # Save the concatenated frames to the list
            concatenated_frames_list.append(clustered_frames)

        # Show side-by-side comparison of original and predicted frames
        comparison = np.concatenate((frames[i], frames[i + 1]), axis=1)
        comparison = cv2.putText(comparison, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Original vs Predicted', comparison)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release memory for the current batch
    del features

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Concatenate frames from the list horizontally for each algorithm
concatenated_frames_algorithm_wise = [np.concatenate(concatenated_frames_list[i:i + len(algorithms)], axis=0) for i in range(0, len(concatenated_frames_list), len(algorithms))]

# Concatenate frames from the list vertically to create a single image
final_image = np.concatenate(concatenated_frames_algorithm_wise, axis=1)

# Save the final image
cv2.imwrite('comparison_of_frames_with_detection_and_clustering_grid.png', final_image)