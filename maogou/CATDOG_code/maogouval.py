import os, cv2, re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import keras.utils.np_utils as np_utils
from tensorflow.keras.applications.resnet50 import ResNet50
import pandas as pd
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

img_width = 150
img_height = 150
data_dir = './train/'
train_images = [data_dir + i for i in os.listdir(data_dir)]
model_save_path = '/home/amax/zxp/wll/maogou/model.h5'

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
train_images = train_images[0:25000]
train_images = np.array(train_images)
X, Y = prepare_data(train_images)
X = np.array(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.validation_data=(x_train,y_train)
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()##.model
        val_targ = self.validation_data[1]###.model
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average=None)###
        _val_precision = precision_score(val_targ, val_predict,average=None)###
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: %f — val_precision: %f — val_recall: %f" %(_val_f1, _val_precision, _val_recall))
        print("— val_f1: %f "%_val_f1)
        return
        


model = ResNet50(input_shape = (150,150,3), weights=None, classes=2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
f1=Metrics()
hist=model.fit(x_train,y_train,batch_size=32,epochs=20,verbose=1,validation_data=(x_train,y_train),callbacks=[f1])
model.evaluate(x_test, y_test)
model.save(model_save_path)