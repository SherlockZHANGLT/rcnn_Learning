import ResNet50
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras.utils.np_utils as np_utils
import tempfile
import numpy as np
import math


def DataSet():
    (x_Train, y_Train), (x_Test, y_Test) = cifar10.load_data()
    x_Train4D = x_Train.reshape(x_Train.shape[0], 32, 32, 3).astype('float32')
    x_Test4D = x_Test.reshape(x_Test.shape[0], 32, 32, 3).astype('float32')
    # 归一化
    x_Train4D_normalize = x_Train4D / 255
    x_Test4D_normalize = x_Test4D / 255
    # one-hot Encoding
    y_TrainOneHot = np_utils.to_categorical(y_Train).astype('float32')
    y_TestOneHot = np_utils.to_categorical(y_Test).astype('float32')
    return [x_Train4D_normalize,y_TrainOneHot,x_Test4D_normalize,y_TestOneHot]


def main(_):
    model = ResNet50(input_shape = (32,32,3), weights=None, classes=10)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    x_Train, y_Train, x_Test, y_Test = DataSet()
    print('x_Train shape : ', x_Train.shape)
    print('y_Train shape : ', y_Train.shape)
    print('x_Test shape : ', x_Test.shape)
    print('y_Test shape : ', y_Test.shape)
    training = model.fit(x_Train, y_Train, epochs=30, batch_size=32)
    plt.plot(training.history['acc'])
    plt.plot(training.history['loss'])
    plt.title('model accuracy and loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.show()
    model.evaluate(x_Test, y_Test)

if __name__ == "__main__":
    tf.app.run(main=main)

