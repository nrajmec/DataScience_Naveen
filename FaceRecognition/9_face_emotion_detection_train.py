from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import seaborn as sns

# Dataset from this link https://www.kaggle.com/msambare/fer2013

IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 32

train_data_dir = 'archive/train'
validation_data_dir = 'archive/test'

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=30,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(train_data_dir,
                                              color_mode='grayscale',
                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)

validation_gen = train_datagen.flow_from_directory(validation_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# img, label = train_gen.__next__()
#
# i = random.randint(0, (img.shape[0]) - 1)
# image = img[i]
# labl = class_labels[label[i].argmax()]
# plt.imshow(image[:, :, 0], cmap='gray')
# plt.title(labl)
# plt.show()

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

num_train_imgs = 0
for root, dirs, files in os.walk(train_data_dir):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(validation_data_dir):
    num_test_imgs += len(files)

epochs = 50

history = model.fit(train_gen,
                    steps_per_epoch=num_train_imgs // batch_size,
                    epochs=epochs,
                    validation_data=validation_gen,
                    validation_steps=num_test_imgs // batch_size)

model.save('emotion_detection_model_100epochs.h5')

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test the model
my_model = load_model('emotion_detection_model_100epochs.h5', compile=False)

# Generate a batch of images
test_img, test_lbl = validation_gen.__next__()
predictions = my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

print("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, predictions)

sns.heatmap(cm, annot=True)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# Check results on a few select images
n = random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_label = class_labels[test_labels[n]]
pred_label = class_labels[predictions[n]]
plt.imshow(image[:, :, 0], cmap='gray')
plt.title("Original label is:" + orig_label + " Predicted is: " + pred_label)
plt.show()
