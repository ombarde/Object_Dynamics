import cv2
import numpy as np
from sklearn.cluster import DBSCAN
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

# Load video from URL (you can replace it with your own video file)
video_path = 'moving_man.mp4'
cap = cv2.VideoCapture(video_path)

# Parameters
num_frames = 5
object_roi = None
eps = 1.5  # Epsilon for DBSCAN
min_samples = 3  # Minimum number of samples for DBSCAN
batch_size = 5

# Process video in batches
num_batches = num_frames // batch_size

# Train the linear regression model
linear_model = LinearRegression()

# Load HOG descriptor and SVM detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    # Check if features are not empty and there are enough frames for training
    if len(frames) > 1 and features.size > 0:
        # Use DBSCAN for prediction
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        predicted_labels = dbscan.fit_predict(features.reshape(features.shape[0], -1))

        # Train a linear regression model for further prediction
        X_train, y_train = features[:-1].reshape(-1, 5 * 5 * 2), features[1:].reshape(-1, 5 * 5 * 2)
        linear_model.fit(X_train, y_train)

    # Display the tracked object using DBSCAN and linear regression predictions
    for i in range(len(frames)):
        if i == len(frames) - 1:
            break

        # Get the label of the last frame using DBSCAN
        dbscan_label = predicted_labels[-1]

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Apply HOG detector to find people in the frame
        found, w = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

        # Draw rectangles around the detected objects
        for (x, y, w, h) in found:
            cv2.rectangle(frames[i], (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show side-by-side comparison of original and predicted frames
        comparison = np.concatenate((frames[i], frames[i + 1]), axis=1)
        cv2.imshow('Original vs Predicted', comparison)

        # Save the concatenated frames to the list
        concatenated_frames_list.append(comparison)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release memory for the current batch
    del features

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Concatenate frames from the list vertically to create a single image
final_image = np.concatenate(concatenated_frames_list, axis=0)

# Save the final image
cv2.imwrite('original_vs_predicted_frames_dbscan.png', final_image)