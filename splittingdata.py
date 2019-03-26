import os
import random
# Splits data into train, test, and validate
orig_images = "C:\\Users\\marcu\\learning\\cleaning\\dataset-original"
final_loc = "C:\\Users\\marcu\\learning\\data"
boxes = {
        "paper": "black",
        "plastic": "blue",
        "glass": "blue",
        "metal": "blue",
        "cardboard": "black"}
#for folder in os.listdir(orig_images):
#    cat = boxes[folder]
#    for file in os.listdir(orig_images + "\\" + folder):
#        select = random.randint(0,100)
#        if select < 70:
#            os.rename(orig_images + "\\" + folder + "\\" + file, final_loc + "\\train\\"+ cat + "\\" + file)
#        else:
#            os.rename(orig_images + "\\" + folder + "\\" + file, final_loc + "\\test\\" + cat + "\\" + file)
#        print(file)
minimum = 997

for folder in os.listdir(final_loc):
    for category in os.listdir(final_loc + "\\" + folder):
        while len(os.listdir(final_loc + "\\" + folder+"\\"+category))>minimum:
            listed = os.listdir(final_loc + "\\" + folder+ "\\" + category)
            index = random.randint(0,len(os.listdir(final_loc + "\\" + folder+"\\"+category)))
            os.remove(final_loc + "\\" + folder+"\\"+category+"\\"+listed[index])