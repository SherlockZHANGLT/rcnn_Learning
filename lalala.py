import os, cv2, re
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
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
#train_images = train_images[0:1300] + train_images[12500:13800]
train_images = np.array(train_images)
X, Y = prepare_data(train_images)
X = np.array(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
datagen = ImageDataGenerator(
    featurewise_center=True,        #将输入数据的均值设置为 0，逐特征进行。
    featurewise_std_normalization=True,     #将输入除以数据标准差，逐特征进行。
    rotation_range=20,                  #随机旋转的度数范围。
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True            #随机水平翻转。
    )
datagen.fit(x_train)
model = ResNet50(input_shape = (150,150,3), weights=None, classes=2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
training = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=32)
'''
plt.plot(training.history['acc'])
plt.plot(training.history['loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'loss'], loc='upper left')
plt.show()
model.evaluate(x_test, y_test)
'''
model.save(model_save_path)