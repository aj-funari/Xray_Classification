# Xray_Classification
Classifying COVID-19 images using AlexNet and ResNet

## Importing Data
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
Run importData.py to download and preprocess data for training. Follow the steps below to 
1. Create a Kaggle account
2. Create an API token
3. Move token to .kaggle local foler

## Data Preprocessing
Once the data is downloaded, delete all files + folders except Normal folder and COVID folder
- The code will not run properly if this step is voided
- We are only interested in classifying between Normal Xray and COVID Xray images. Make sure to remove the masks image folders inside the Normal and COVID folders.
- At this point, there are about 10,192 Normal images and 3,616 COVID images. Training this data will overfit the model to always predicting Normal results.
- Copy and paste the COVID images three times. This wil validate there are roughly the same Normal/COVID images to prevent overfitting.
