# Distracted Driver Detection

## Overview
This project, developed by Shimran Priyadarshini, uses a ResNet-50 model to detect distracted driving behaviors from images in the State Farm Distracted Driver Detection dataset (Kaggle), classifying them into 10 categories like safe driving or texting.

## Dataset
- **Source**: Kaggle State Farm Distracted Driver Detection
- **Size**: 22,424 labeled images
- **Files**:
  - `driver_imgs_list.csv`: Contains image metadata (subject IDs, class labels)
  - `imgs.zip`: Train/test image folders

## Model
- **Architecture**: ResNet-50
- **Training Parameters**:
  - Epochs: 10
  - Batch Size: 32
  - Optimizer: Adam
  - Initializer: Glorot Uniform
- **Performance**:
  - Training Loss: 0.93
  - Validation Loss: 3.79
  - Holdout Loss: 2.64

## Features
- **Preprocessing**: Images resized, normalized, and saved as arrays (`CreateImgArray`, `Rescale`)
- **EDA**: Class distribution analysis (`PlotClassFrequency`, `DescribeImageData`)
- **Evaluation**: Leave-One-Group-Out cross-validation (`LOGO`)
- **Visualization**: Image display with predictions (`PrintImage`)

## Dependencies
- TensorFlow, Keras
- NumPy, Pandas
- Matplotlib, Scikit-learn

## Usage
1. Clone repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Jupyter Notebook: `jupyter notebook Distracted_Driver_detection.ipynb`

## Challenges
- High losses due to limited resources for hyperparameter tuning
- Future Work: Enhance with grid search optimization

## License
MIT License
