from __future__ import print_function, division
from builtins import range, input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from util import plot_confusion_matrix

# Function to prepare the generators data for the confusion matrix sklearn function
# and create and return the confusion matrix
def get_confusion_matrix(data_path, N):
    # We need to see the data in the same order
    # for batch predictions and targets
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        # The generator runs in an infinite loop, so we need to exit manually
        if len(targets) >= N:
            break
    cm = confusion_matrix(targets, predictions)
    return cm

# Resize all the images to 100x100
IMAGE_SIZE = [100, 100]

# Setting the training configuration
epochs = 2
batch_size = 32

# Setting paths
#train_path = "../../data/fruits-360-small/Training"
#valid_path = "../../data/fruits-360-small/Validation"
train_path = "../../data/fruits-360/Training"
valid_path = "../../data/fruits-360/Test"

# Util for getting the number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# Util for getting the number of classes
folders = glob(train_path + '/*')

# Looking at an image...
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

# Add the preprocessing layer to the front of VGG
# weights='imagenet' so that we can use VGG with pre-traine weights on imagenet
# include_top is set to false because we want to train the last layer for our data
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# We only want to train the last layer so we do not train the existing VGG weights
for layer in vgg.layers:
    layer.trainable = False

# The layers we add in order to perform transfer learning, the last layer
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)  # We could add more hidden layers...
prediction = Dense(len(folders), activation='softmax')(x)

# Creating a model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
model.summary()

# Set the cost and optimization method to use in the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Defining an instance of the ImageDataGenerator
gen = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         preprocessing_function=preprocess_input)

# Test operator to see how it works and some other useful things

# Get label mapping for confusion matrix plot
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)

# Creating a dictionary that maps from the class name to its index
# Key is the class name
# Value is the class index
for k, v in test_gen.class_indices.items():
    labels[v] = k

# Verifying that the data has been processed correctly
# with keras "preprocessing_function" function
# also the resizing, the order of the color bands because VGG was
# trained using Cafe and Cafe doesn't use RGB, it uses BGR
# also the substraction of the mean pixel intensities from each color band
for x, y in test_gen:
    print("min:", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

# Creating the generators
train_generator = gen.flow_from_directory(train_path,
                                          target_size=IMAGE_SIZE,
                                          shuffle=True,
                                          batch_size=batch_size)
valid_generator = gen.flow_from_directory(valid_path,
                                          target_size=IMAGE_SIZE,
                                          shuffle=True,
                                          batch_size=batch_size)

# Fitting the model
r = model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        epochs=epochs,
                        steps_per_epoch=len(image_files) // batch_size,
                        validation_steps=len(valid_image_files) // batch_size)

# Creating the confusion mmatrix
cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

# Plotting some data

# logs
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

plot_confusion_matrix(cm, labels, title='Train Confusion Matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation Confusion Matrix')
