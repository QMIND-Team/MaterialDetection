# =============================================================================
# from keras.preprocessing.image import load_img, ImageDataGenerator
# from keras.preprocessing.image import img_to_array
# from keras import optimizers
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# =============================================================================
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
# load the model
# =============================================================================
# model = VGG16()
# # load an image from file
# #throw the file name of the picture below and make sure its in the right directory
# os.chdir("mats")
# for file in os.listdir():
#     print('Working on %s' % file)
#     image = load_img(file, target_size=(224, 224))
#     # convert the image pixels to a numpy array
#     image = img_to_array(image)
#     # reshape data for the model
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # prepare the image for the VGG model
#     image = preprocess_input(image)
#     # predict the probability across all output classes
#     yhat = model.predict(image)
#     # convert the probabilities to class labels
#     label = decode_predictions(yhat)
#     # retrieve the most likely result, e.g. highest probability
#     label = label[0][0]
#     # print the classification
#     print('%s (%.2f%%)' % (label[1], label[2]*100))
# =============================================================================

# Hey Leon! You're going to have to tweak a couple of things really quick before running this
# Throw in the path directory to the materials file I sent you:
train_loc = "C:\\Users\\marcu\\learning\\mats\\train"
test_loc = "C:\\Users\\marcu\\learning\\mats\\test"
my_base = VGG16(weights='imagenet',include_top = False, input_shape = (224,224,3))
my_base.trainable = False
model = models.Sequential()
model.add(my_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(5,activation = 'softmax'))

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 180,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0,
        zoom_range = .2,
        horizontal_flip = True,
        fill_mode = 'nearest'
        )
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_loc,
        target_size = (224,224),
        batch_size=20,
        class_mode='categorical'
        )
valid_generator = test_datagen.flow_from_directory(
        test_loc,
        target_size = (224,224),
        batch_size = 20,
        class_mode = 'categorical')
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='modelWeights/best_weights_VGG16.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 5,
        validation_data = valid_generator,
        callbacks = callbacks;
        validation_steps = 50)
