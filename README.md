
# 🎥 Video Object Detection & Spatio-Temporal Clustering

This project demonstrates object detection and motion-based frame clustering from a video using multiple machine learning and computer vision algorithms.
## 📌 Overview

The system performs the following tasks:
- Extracts spatio-temporal features using optical flow
- Detects people using HOG + SVM
- Applies multiple clustering algorithms: 
  - Gaussian Mixture Model (GMM)
  - MeanShift
  - DBSCAN
  - KMeans
- Visualizes and compares the results side-by-side
- Saves annotated comparisons of original and predicted frames
- Generates a final summary visualization and report

## 📂 Project Structure

```
├── moving_man.mp4             # Sample video file (not included in repo)
├── video_analysis.py          # Main Python script
├── results/
│   ├── hog_detection/         # HOG + SVM results
│   ├── clustering_comparisons/ # GMM, DBSCAN, MeanShift, KMeans results
│   └── final_summary.png      # Combined image of all algorithms
├── report/
│   ├── visual_report.png      # Visual report
│   └── analysis_report.md     # Detailed analysis writeup
└── README.md
```

## 🚀 Installation

Make sure you have Python 3.8+ and install the required libraries:

```bash
pip install numpy opencv-python scikit-learn matplotlib scipy
```

## 🔍 How It Works

1. **Object Detection:** HOG + SVM detects people in each frame.
2. **Optical Flow:** Extracts motion vectors between frames using Farneback algorithm.
3. **Clustering:** Applies GMM, MeanShift, DBSCAN, and KMeans to group motion patterns.
4. **Comparison:** Saves side-by-side images of the original vs predicted clusters per algorithm.
5. **Final Image:** Combines all algorithm results into a single image with labels.
6. **Reporting:** A markdown and image report summarize findings and algorithm performance.

## 📊 Algorithms Compared

| Algorithm | Strengths | Limitations |
|----------|-----------|-------------|
| **GMM** | Models complex distributions | Sensitive to initial parameters |
| **MeanShift** | No need to specify number of clusters | Computationally expensive |
| **DBSCAN** | Handles noise and density well | Struggles with varying densities |
| **KMeans** | Fast and simple | Assumes spherical clusters |
| **HOG + SVM** | Accurate human detection | Limited to rigid body patterns |


## 📝 Academic Note

This project is part of a major academic submission and holds high importance in the final evaluation. Kindly avoid reproducing or distributing it without proper credits.
