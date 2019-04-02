import cv2
import numpy as np
import time
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Our decoder dictionary translates VGG16 category predictions to our categories.
# This fun'n returns our decoder dictionary.
# 1 represents blue box items and 0 represents black box items.
def getDecode():
    decoder = dict(
        carton=0,
        pencil_sharpener=0,
        envelope=0,
        packet=0,  # need to change?
        switch=1,
        paper_towel=0,
        Band_Aid=0,
        toilet_tissue=0,
        birdhouse=0,
        water_bottle=1,
        pill_bottle=1,
        nipple=1,
        rubber_eraser=1,
        pop_bottle=1,
        spotlight=1,
        piggy_bank=1,
        whistle=1,
        screwdriver=1,
        wine_bottle=1,
        syringe=1,
        lighter=1,
        bottlecap=1,
        oil_filter=1,
        beer_bottle=1,
        thimble=1,
        ballpoint=1,
        shower_cap=1,
        handkerchief=0,
        Christmas_stocking=0,
        wallet=0,
        pencil_box=0,
        binder=0,
        menu=0,
    )
    return decoder


# Captures a picture using the Picamera and returns the original image with a 224x224 pixel image tat our model can use.
def getPicture(camera):
    return_value, image = camera.read()
    imageResize = np.array(cv2.resize(image, (224, 224)))
    input = np.zeros((1, 224, 224, 3))
    input[0] = imageResize
    del (camera)
    return input, image

# Runs the VGG16 model and converts predictions to one of our 3 results: blue, black or unknown.
def getPrediction(model, image, decoder):
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    prediction = 2
    for i in range(0, 5):
        try:
            prediction = decoder[label[0][i][1]]
            break
        except:
            prediction = 2
    return prediction

# Runs the camera
def showVid(VGG16_Model):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    decoder = getDecode()

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 80:  # exit on ESC
            break
        elif key == 27: # press p (80) for picture
            input, raw_image = getPicture(vc)
            results = getPrediction(VGG16_Model, input, decoder)
            if results == 0:
                text = "Black Box"
            elif results == 1:
                text = "Blue Box"
            else:
                text = "Unknown"
            cv2.putText(raw_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow("preview", raw_image)
            cv2.waitKey(0)
    vc.release()
    #cv2.destroyWindow("preview")

# Load in the VGG16 model from keras
print("Loading Keras' VGG16 model.")
model = VGG16()
# Load in the decoder to translate VGG16 predictions to our predictions
showVid(model)