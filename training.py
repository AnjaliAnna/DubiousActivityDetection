# preprocessing and training
import cv2  # computer vision library for image processing
import math 
import matplotlib.pyplot as plt # for plotting or plot generation
import pandas as pd # allows importing data from various file formats like csv
import tensorflow as tf # for building model
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize

data = pd.read_csv(image.csv)
print(data.head(5))
x = []
print("hello")
