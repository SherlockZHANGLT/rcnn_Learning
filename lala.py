import os, cv2, re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import keras.utils.np_utils as np_utils
from tensorflow.keras.applications.resnet50 import ResNet50

img_width = 150
img_height = 150
data_dir = './input/train/'
train_images = [data_dir + i for i in os.listdir(data_dir)]
model_save_path = './'


def prepare_data(list_of_images):
    x = []  # images
    y = []  # labels
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width, img_height), interpolation=cv2.INTER_CUBIC))
    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
    y = np_utils.to_categorical(y).astype('float32')
    return x, y

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

train_images.sort(key=natural_keys)
train_images = train_images[0:1300] + train_images[12500:13800]
train_images = np.array(train_images)
X, Y = prepare_data(train_images)
X = np.array(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model = ResNet50(input_shape = (150,150,3), weights=None, classes=2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
training = model.fit(x_train, y_train, epochs=5, batch_size=32)
plt.plot(training.history['acc'])
plt.plot(training.history['loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'loss'], loc='upper left')
plt.show()
model.evaluate(x_test, y_test)
