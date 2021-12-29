import cv2  # for capturing videos
import math  # for mathematical operations
import os  # functions to interact with os

path = 'G:/miniproject/reference/video/input'

files = []

for f in os.listdir(
        path):  # returns a list containing the names of the entries in the directory given by path ie 'input'
    files.append(f)  # appends an element to the end of the list
# print files

count = 0

for i in range(0, len(files)):
    videoFile = 'video/input/' + str(files[i])  # VideoFile contains path of the video
    # print videoFile
    cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    x = 1
    while cap.isOpened():  # while videofile is opened
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        # Basically, ret is a boolean regarding whether or not there was a return at all,
        # at the frame is each frame that is returned.
        # If there is no frame, you wont get an error, you will get None.
        if ret != True:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = "image/image%d.jpg" % count;  # naming the file
            count += 1
            cv2.imwrite(filename, frame)
            # used to save an image to a storage device
            # returns true if image is saved successfully
    cap.release()

print("Done!")
