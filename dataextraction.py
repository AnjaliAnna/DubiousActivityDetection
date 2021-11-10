# put a video as input and extract from it a frame every second keeping the original size and saving all the frames
# into a folder
import cv2.cv2 as cv2  # opencv for image processing
import math  # math module defines mathematical functions like trig log etc
import os  # contains functions for creating and removing directories fetching contents etc

path = "video/input"
files = []
for f in os.listdir(path):
    # listdir() returns list of all files in specified directory ie. input
    files.append(f)
    # print(files) gives ['harm1.mpg', 'harm2.mpg', 'harm3.mpg', 'harm4.mpg', 'pot1.mpg', 'pot10.mpg', 'pot2.mpg',
    # 'pot3.mpg', 'pot4.mpg', 'pot5.mpg', 'pot6.mpg', 'pot7.mpg', 'pot8.mpg', 'pot9.mpg', 'safe1.mpg', 'safe10.mpg',
    # 'safe11.mpg', 'safe12.mpg', 'safe13.mpg', 'safe2.mpg', 'safe3.mpg', 'safe4.mpg', 'safe5.mpg', 'safe6.mpg',
    # 'safe7.mpg', 'safe8.mpg', 'safe9.mpg']
count = 0
for i in range(0, len(files)):
    videoFile = 'video/input/' + str(files[i])
    # print(videoFile)  videoFile contains path to video files in input folder
    # ie. video/input/safe9.mpg  video/input/safe8.mpg etc
    capt = cv2.VideoCapture(videoFile)  # captures video file in given path
    frameRate = capt.get(5)
    x = 1
    while capt.isOpened():
        frameId = capt.get(1)  # get() method reads metadata of video
        # get(1) returns the current frame number
        ret, frame = capt.read()
        # read() method is used inside the loop to read one frame at a time from the video stream read() returns a
        # tuple of two values ret stores the boolean returned ie true if there is a frame false otherwise frame
        # stores the actual video frame

        if not ret:
            break  # if no more frame to read

        if frameId % math.floor(frameRate) == 0:
            filename = "image/image%d.jpg" % count  # file path of frame to be stored
            count += 1
            cv2.imwrite(filename, frame)  # imwrite() saves images in specified path
    capt.release()

print("Data Extraction Done!")
