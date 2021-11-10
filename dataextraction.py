# importing modules
import cv2.cv2 as cv2  # opencv for image processing
import math  # math module defines mathematical functions like trig log etc
import os  # contains functions for creating and removing directories fetching contents etc

path = "G:/miniproject/dad/DubiousActivityDetection/video/input"
files = []
for f in os.listdir(path):
    # listdir() returns list of all files in specified directory ie. input
    files.append(f)
# print(files) gives
# ['harm1.mpg', 'harm2.mpg', 'harm3.mpg', 'harm4.mpg', 'pot1.mpg', 'pot10.mpg', 'pot2.mpg', 'pot3.mpg', 'pot4.mpg',
# 'pot5.mpg', 'pot6.mpg', 'pot7.mpg', 'pot8.mpg', 'pot9.mpg', 'safe1.mpg', 'safe10.mpg', 'safe11.mpg', 'safe12.mpg',
# 'safe13.mpg', 'safe2.mpg', 'safe3.mpg', 'safe4.mpg', 'safe5.mpg', 'safe6.mpg', 'safe7.mpg', 'safe8.mpg', 'safe9.mpg']
count = 0
for i in range(0, len(files)):
    videoFile = 'video/input/' + str(files[i])
    # print(videoFile)  videoFile contains path to video files
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5)
    x = 1
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = "image/image%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()

    print("Done!")
