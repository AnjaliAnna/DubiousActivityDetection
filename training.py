# preprocessing and training
import cv2  # computer vision library for image processing
import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize

data = pd.read_csv(image.csv)
print(data.head(5))
x = []
