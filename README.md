# WeedDetector
Model that trains weed detection through a dataset and saved the model as pre-trained weights in the .h5 format. Uses keras, tensorflow, and M_CRNN

---
![alt text](https://i.imgur.com/sHqbMdb.png "Weed Detection Image")

## WeedDetector.py
Weeddetector.py is the main script. This script helped to train the model. It contains the WeedDetector Class that helps train the model and saves the model once it is trained as an .h5 file.

## PretrainedWeightUrl.txt
This is the actual .h5 (Hierarchical Data Format 5 File), which acts as pre-trained weights so that anyone can use the model that I trained. Because the pre-trained weights are a very large file size, GitHub doesn't support uploading these files. As a result, I uploaded the .h5 onto mediafire for anyone to use. 

## Weed
The weed folder is a folder that contains a sample of the dataset used to train the model. For each image in the folder, there is a respective .XML file. Upon viewing any of the .XML files, you can see that there are bounding boxes for each weed, with minimum and maximum x and y values.

## Testing.py
Script that shows that the model works, and is the example used in video.

## M_CRNN Utilities
These are neccessary utilities from M_CRNN that allow us to easily train our model with image segmentation. This includes assets, build, dist, mask_rcnn-egg-info, and mrcnn.

## Setup.py
Setup.py is a script that helps to setup all the required M_CRNN utilities into the folder that setup.py is within.

## requirements.txt
Requirements.txt is a list of all the python libraries utilized.

## Prediction.py
Used to test the already created model using the pre-trained weights (model_path = '.h5')

## image.txt
Random image making sure false weeds are not detected. 






