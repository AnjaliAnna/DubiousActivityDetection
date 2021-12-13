import cv2  # for capturing videos
import math  # for mathematical operations
import os

path = 'G:/miniproject/reference/video/input'

files = []

for f in os.listdir(path):
    files.append(f)
# print files

count = 0

for i in range(0, len(files)):
    videoFile = 'video/input/' + str(files[i])
    # print videoFile
    cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    x = 1
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = "image/image%d.jpg" % count;
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()

print("Done!")
