# LEGO Object Detection with YOLOv5

This project focuses on detecting LEGO pieces using a custom-trained YOLOv5 model. The dataset was sourced from Kaggle, and due to resource constraints, a subset of 100 images was used for this implementation. The project involves data preprocessing, format conversion, model training, evaluation, and visualization of results.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [Conclusion](#conclusion)

---

## Overview
This project demonstrates object detection capabilities by leveraging YOLOv5 (You Only Look Once), a real-time object detection model. By detecting LEGO pieces within images, the model showcases how bounding boxes and confidence scores can be generated for object localization.

Key components include:
- Converting the dataset from Pascal VOC format to YOLO format.
- Resizing images and adjusting bounding boxes.
- Training the YOLOv5 model on a subset of 100 images.
- Evaluating performance using precision, recall, and mAP.
- Visualizing predictions on test images.

---

## Dataset
- **Source**: Kaggle [LEGO dataset](https://www.kaggle.com/datasets/dreamfactor/biggest-lego-dataset-600-parts)
- **Subset Used**: 100 images selected for this project due to resource limitations.
- **Format**: Original dataset was in Pascal VOC XML format.

---

## Data Preparation
1. **Format Conversion**: 
   - The Pascal VOC annotations (XML) were converted to YOLO format. 
   - YOLO format uses class labels and normalized bounding box coordinates ([class, x_center, y_center, width, height]).

2. **Image Resizing**:
   - Images were resized to 640x640 pixels for compatibility with YOLOv5.
   - Bounding box coordinates were adjusted proportionally to match the resized images.

3. **Directory Structure**:
   - Data was split into `train`, `val`, and `test` subsets with corresponding images and labels.

4. **Configuration**:
   - A `dataset.yaml` file was created to define paths, class names, and number of classes.

---

## Model Training
- **Model**: YOLOv5
- **Epochs**: 50
- **Batch Size**: 8
- **Learning Rate**: 0.0005
- **Optimizer**: Adam
- **Input Size**: 640x640

### Steps:
1. Loaded the YOLOv5 model.
2. Configured the dataset and training parameters.
3. Trained the model using the prepared training and validation datasets.

---

## Evaluation
The model's performance was evaluated using the following metrics:

- **Precision**: Measures how many predicted bounding boxes are correct.
- **Recall**: Measures how many ground truth objects were detected.
- **mAP@0.5** (Mean Average Precision at IoU=0.5): Overall detection accuracy.
- **mAP@0.5:0.95**: Measures accuracy across IoU thresholds from 0.5 to 0.95.

### Results:
| Metric         | Value       |
|----------------|-------------|
| Precision      | 93.35%      |
| Recall         | 90.38%      |
| mAP@0.5        | 95.10%      |
| mAP@0.5:0.95   | 85.27%      |

---

## Inference
The model was tested on unseen images to evaluate its performance.
- Images and predictions were displayed with bounding boxes (green for ground truth, red for predictions).
- Confidence scores were included to highlight prediction confidence.

---

## Results
- The model successfully detected LEGO pieces with high precision and recall.
- Bounding boxes closely aligned with ground truth annotations.
- Overlapping predictions were reduced by applying a confidence threshold of 0.5.

---

## How to Run
1. Clone the repository.
   ```bash
   git clone <repo-link>
   cd <project-folder>
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset (ensure it follows the YOLOv5 format).

4. Train the model.
   ```bash
   python train.py --data dataset.yaml --epochs 50 --batch 8 --img 640
   ```

5. Run inference.
   ```bash
   python detect.py --weights path/to/weights --source path/to/test/images --conf 0.5
   ```

---

## Conclusion
This project demonstrates the successful implementation of YOLOv5 for object detection on a subset of the LEGO dataset. Despite resource constraints, the model achieved strong performance with a high mAP score, showcasing YOLO's efficiency and accuracy for detecting objects in small-scale datasets. Further improvements can be made by expanding the dataset, fine-tuning hyperparameters, and addressing edge cases like overlapping predictions.

---

**Future Work**:
- Train on a larger subset of the dataset.
- Experiment with other YOLO versions (YOLOv6, YOLOv8).
- Implement post-processing techniques to further reduce overlaps.
