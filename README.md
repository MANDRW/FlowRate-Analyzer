# FlowRate Analyzer – CNN-Based Classification of Extrusion Flow Ratio in FDM 3D Printing

A system for classifying flow-ratio calibration blocks in FDM 3D printing using convolutional neural networks.

## About

FlowRate Analyzer is a machine-learning project developed for an engineering thesis.  
The system analyzes images of 3D-printed calibration blocks and classifies them into three categories:

- **Under-extrude**  
- **Optimal extrusion**  
- **Over-extrude**

The project includes a complete pipeline: dataset preparation, model training, evaluation, and comparison of architectures.

## Dataset

*(In the future, this section will link to a public dataset on Kaggle.)*

## Features

- **Image Preprocessing:** cropping, resizing, normalization, augmentation  
- **Two Neural Network Architectures:**  
  - Custom CNN  
  - MobileNetV2 (transfer learning)  
- **Cross-validation:** Stratified K-Fold for reliable model evaluation
  
## Technology Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Data Handling:** NumPy, PIL, scikit-learn  
- **Visualization:** Matplotlib  
- **Environment:** CUDA-accelerated GPU training (NVIDIA)

## Author

**Mateusz Andrzejewski**  
Engineering Thesis Project (2025)


