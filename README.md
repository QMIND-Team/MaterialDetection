# MaterialDetection
VGG16 CNN with dense top trained on subset of data
The user will have to update train_loc and test_loc with the file path to the train and test folders.
The reduced data set is available through Google Drive at:
https://drive.google.com/open?id=1mnK58jWv5jKFwKLnVCkm4lHMUkkdGAys.
It should already be in the correct format, so just download it.

VGG_demo.py:
Holds a pretrained VGG16 network that can be used to classify an object, edit line 11 with desired directory

VGG_try1.py:
1st attempt at a fine tuned VGG16 network, froze all convolutional weights and only working with the dense layer on top
