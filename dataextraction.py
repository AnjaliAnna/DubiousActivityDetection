# importing modules
import cv2
import numpy as np
import math  # math module defines mathematical functions like trig log etc
import os  # contains functions for creating and removing directories fetching contents etc

path = "G:/miniproject/dad/DubiousActivityDetection/video/input"
files = []
for f in os.listdir(path):
    # listdir() returns list of all files in specified directory ie. input
    files.append(f)
