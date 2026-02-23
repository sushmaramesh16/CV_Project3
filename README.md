# Name: Sushma Ramesh, Dina Barua
# Date: February 22, 2026
# Purpose: README file for Project 3 - Real-time 2D Object Recognition

---

## Authors
- Sushma Ramesh
- Dina Barua

---

## Demo Video
https://drive.google.com/file/d/1Ew6TP9fqHumvHHZnFrPFO6zK1jHp66rN/view?usp=sharing

---

## Operating System & IDE
- **OS:** macOS
- **IDE:** VS Code
- **Language:** Python 3.13

---

## Project Description
This project implements a 2D object recognition system that identifies objects placed on a
uniform white paper background in a translation, scale, and rotation invariant manner.
The system processes still images and directories of images rather than live webcam input,
but is structured to work frame-by-frame so it could be extended to live video with minimal
changes.

The full pipeline consists of:
1. Dynamic HSV thresholding using a custom ISODATA algorithm written from scratch
2. Morphological filtering using erosion and dilation written from scratch in NumPy
3. Connected components analysis to segment the image into labelled regions
4. Feature extraction using Hu moments, aspect ratio, extent, and solidity
5. Nearest-neighbour classification using scaled Euclidean distance
6. Performance evaluation using a 5x5 confusion matrix
7. One-shot eigenspace classification using PCA-based image embeddings

The five objects recognized are:
- Set square (triangle)
- T-bar hex key (tbar)
- L-shaped allen key (lkey)
- Wood chisel (chisel)
- Car remote key (carkey)

All five objects were correctly classified on the dev image set achieving 100% accuracy
with both the hand-built feature classifier and the eigenspace classifier.

---

## Dependencies
Install all required packages before running:
```
pip install opencv-python numpy matplotlib
```
Python version: 3.13
No additional setup or environment variables are required.

---

## File Structure
```
CV_PROJECT3/
├── task1_threshold.py       # Task 1: Dynamic HSV thresholding (ISODATA, written from scratch)
├── task2_morphology.py      # Task 2: Morphological open/close (written from scratch in NumPy)
├── task3_components.py      # Task 3: Connected components analysis and region coloring
├── task4_orientation.py     # Task 4: Feature computation, oriented bounding box, principal axis
├── task5_training.py        # Task 5: Training data collection and object_db.json management
├── task6_classify.py        # Task 6: Nearest-neighbour classification with dual distance metrics
├── task7_evaluate.py        # Task 7: Confusion matrix evaluation across all objects
├── task9_eigenspace.py      # Task 9: One-shot PCA eigenspace classification
├── extension_unknown.py     # Extension 1: Unknown object detection via distance thresholding
├── object_db.json           # Training database storing feature vectors and labels
├── eigenspace_db.npz        # Stored eigenspace: mean image and eigenvectors for projection
├── IMGS/                    # Input images used for training and testing
└── outputs/                 # All result images, classified outputs, and confusion matrices
```

---

## How to Run

### Task 1 – Thresholding
Applies dynamic HSV ISODATA thresholding to separate objects from the white background.
Displays four panels: Original, V channel, S channel, and Binary mask.
The ISODATA algorithm runs on both the V and S channels independently and uses an
OR-combination so both dark and colourful objects are detected correctly.
```
python task1_threshold.py --image IMGS/<image_file>
python task1_threshold.py --dir IMGS/
```

### Task 2 – Morphological Filtering
Applies morphological open (5x5 kernel) to remove small noise blobs, followed by
morphological close (9x9 kernel) to fill small holes caused by reflections.
Both operations are written from scratch using NumPy sliding window operations —
no cv2.erode, cv2.dilate, or cv2.morphologyEx calls are used.
Displays five panels: Original, Task 1 binary, After Open, After Close, Overlay.
```
python task2_morphology.py --image IMGS/<image_file>
```

### Task 3 – Connected Components
Segments the cleaned binary image into labelled regions using OpenCV
connectedComponentsWithStats with 8-connectivity. Regions smaller than 3000px
or larger than 40% of the image area are discarded. Each valid region is assigned
a color from a fixed palette. Bounding boxes and centroid dots are overlaid.
Displays four panels: Original, Cleaned binary, Color-coded CC map, Bounding boxes.
```
python task3_components.py --image IMGS/<image_file>
```

### Task 4 – Feature Computation
Computes the principal axis orientation from second-order central moments (written
from scratch). Uses cv2.minAreaRect for the oriented bounding box. Extracts a
10-dimensional feature vector: 7 log-scaled Hu moments, aspect ratio, extent,
and solidity. All features are translation, scale, and rotation invariant.
Displays four panels: Original, CC map, Axis + oriented bounding box, Feature annotations.
```
python task4_orientation.py --image IMGS/<image_file>
```

### Task 5 – Training Data Collection

**Interactive mode** — shows the detected object and prompts for a label:
```
python task5_training.py --image IMGS/<image_file>
```
- A window displays the detected region with bounding box and feature values
- Press any key to confirm the detection
- Type the label in the terminal when prompted
- The feature vector and label are appended to object_db.json

**Auto mode** — label is derived automatically from the filename:
```
python task5_training.py --dir IMGS/ --auto
```
For example, triangle_01.png is labelled as "triangle" automatically.
This enables fast batch-labelling of an entire image directory.

### Task 6 – Classification
Classifies each image using nearest-neighbour with scaled Euclidean distance.
The feature vector is compared against every entry in object_db.json.
The label of the closest match is shown on the output image along with the distance value.
Also computes plain Euclidean distance and shows both results side by side for comparison.
```
python task6_classify.py --image IMGS/<image_file>
python task6_classify.py --dir IMGS/
```

### Task 7 – Evaluation
Runs the full pipeline on all images in the directory and compares predicted labels
against ground truth. Builds a 5x5 confusion matrix where rows are true labels and
columns are predicted labels. Saves the confusion matrix plot to outputs/.
```
python task7_evaluate.py --dir IMGS/
```

### Task 8 – Demo Video
No code file for this task. The demo video is linked at the top of this README.
The video shows the terminal output and classification popup window for each object.

### Task 9 – Eigenspace One-Shot Classification
Builds a PCA eigenspace from pre-processed training images. Each object ROI is
rotated to align the primary axis with the X-axis, cropped, resized to 160px on
the long side, and flattened to a 1D vector. SVD is run on the difference matrix
to get eigenvectors. Each image is projected onto the top 6 eigenvectors to get
a 6-dimensional embedding. Classification uses sum-squared difference between
the query embedding and all training embeddings. Achieves 100% on the dev set,
matching the hand-built classifier. Saves confusion matrix to outputs/.
```
python task9_eigenspace.py --dir IMGS/
```

### Extension 1 – Unknown Object Detection
Runs the scaled Euclidean nearest-neighbour classifier. If the best match distance
exceeds a threshold (default 3.0), the object is labelled UNKNOWN in red instead
of being assigned an incorrect category label.
```
python extension_unknown.py --image IMGS/<image_file>
python extension_unknown.py --dir IMGS/
```
To test with an unknown object, use an image of any object not in object_db.json.
The label will appear in red as UNKNOWN with the distance shown.

---

## Extensions

### Extension 1 – Unknown Object Detection
The classifier detects when an object does not belong to any known category.
If the nearest-neighbour scaled Euclidean distance exceeds 3.0, the object is
flagged as UNKNOWN in red. Known objects produce distances near 0.0. This
prevents the system from confidently misclassifying unseen objects.
Implemented in extension_unknown.py.

### Extension 2 – Distance Metric Comparison
Two distance metrics are implemented and compared on every classified image:
- Scaled Euclidean: divides each feature dimension by its training-set standard
  deviation before computing distance, preventing large-valued features from dominating
- Plain Euclidean: no normalisation applied
Both metrics agreed on all five dev images. Scaled Euclidean is expected to be
more robust to lighting variation and unseen objects. Output shows both results
side by side with a checkmark if they agree or DISAGREE if they differ.
Built into task6_classify.py.

---

## Written From Scratch (no OpenCV equivalents used)
- ISODATA dynamic thresholding on V and S channels (task1_threshold.py)
- Morphological erosion using sliding AND across kernel offsets (task2_morphology.py)
- Morphological dilation using sliding OR across kernel offsets (task2_morphology.py)
- Principal axis orientation from second-order central moments (task4_orientation.py)

---

## Time Travel Days
Not using any time travel days — submitting on time.