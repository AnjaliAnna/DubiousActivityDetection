# preprocessing and training
import cv2  # computer vision library for image processing
import math  # mathematical functions
import matplotlib.pyplot as plt  # for plotting or plot generation
import pandas as pd  # allows importing data from various file formats like csv
import tensorflow as tf  # for building model
from keras.preprocessing import image
import numpy as np  # library with tools supporting large multi dimensional arrays
from keras.utils import np_utils  # collection of utilities for array and list mgmt
from skimage.transform import resize  # skimage is collection of algos for image processing

# from transform module in skimage we import resize
# resize returns resized version of the input image

dataf = pd.read_csv('train_img.csv')  # imports a csv file to dataframe format
# print(dataf.head(5))
X = [] #x array
for img_name in dataf.Image_ID:
    img = plt.imread('image/' + img_name + '.jpg', 0)  # reads image from file into an array
    X.append(img)  # storing each image in x array

X = np.array(X)  # converting list to array
print(X.shape)
y = dataf.classes
dummy_y = np_utils.to_categorical(y)
image = []
for i in range(0, X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224, 224)).astype(int)
    image.append(a)
X = np.array(image)

# preprocessing

from keras.applications.vgg16 import preprocess_input

X = preprocess_input(X, mode='tf')

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3,
                                                      random_state=42)  # preparing validation set

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))  # include top=false to remove the top layer

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)

# model building
model = Sequential()
