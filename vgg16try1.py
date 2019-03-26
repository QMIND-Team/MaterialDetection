# =============================================================================
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras import optimizers
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# =============================================================================
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

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
model.add(layers.Dense(1,activation = 'sigmoid'))

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
        class_mode='binary'
        )

valid_generator = test_datagen.flow_from_directory(
        test_loc,
        target_size = (224,224),
        batch_size = 20,
        class_mode = 'binary')

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['accuracy'])


history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_data = valid_generator,
        validation_steps = 50)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")
