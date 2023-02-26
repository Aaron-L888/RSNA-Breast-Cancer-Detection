# RSNA-Breast-Cancer-Detection
#Radiological Society of North America - Mammogram Breast Cancer Detection Model

54,706 Mammogram DICOM files provided by RSNA, with 1153 images positive for cancer. Highly imbalanced dataset. Associated metadata on each image also provided.

Full, planned model was meant to be an ensemble of 4 CNN architectures: Resnet50, InceptionV3, EfficientNetB2, and ConvNeXtSmall. 

Current Model has been produced for resnet and convnext architectures.
DICOM files are handled using the dicomsdl library and converted into pixel array data, and labels are added to the images for training and validation. 
Dense layers are trained first while CNN layers are frozen, then select top CNN layers are unfrozen and full training is completed. 
Predicted outputs are placed in a csv file for review.

