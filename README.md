# Azure Custom Vision for image classification

## 🚀 Overview
Azure Custom Vision enables rapid creation of custom image classifiers using a simple web interface or API. It supports for image classification:
- Multi-class classification: Assigning one label per image.
- Multi-label classification: Assigning multiple labels per image.

## 🛠️ How It Works
- Upload and Tag Images: Use the portal or API to upload training images and assign tags.
- Train the Model: Azure uses transfer learning to train a model based on your dataset.
- Evaluate Performance: Review precision, recall, confusion matrix and accuracy metrics.
- Deploy: Publish the model as a REST API or export it for offline use (Onnx, Docker, TensorFlow, CoreML).

## 📚 Resources
- https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/overview
- https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/getting-started-build-a-classifier

## Notebooks
### Training
- https://github.com/retkowsky/image-classification-azure-custom-vision/blob/main/Image%20classification%20-%20Training%20-%20Azure%20Custom%20Vision.ipynb
### Prediction
- https://github.com/retkowsky/image-classification-azure-custom-vision/blob/main/Image%20classification%20-%20Prediction%20-%20Azure%20Custom%20Vision.ipynb

## Azure Custom Vision
<img src="screenshot.jpg">

## Images and tags
<img src="screenshot1.jpg">

## Model results
<img src="screenshot2.jpg">

## Predictions
<img src="screenshot3.jpg">
