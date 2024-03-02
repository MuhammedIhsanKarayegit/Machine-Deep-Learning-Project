import cv2 as cv
import os
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import dlib as db


cap = cv.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480 ) # set video height

img_background = cv.imread('Resources/background.png')

# Importing the mod images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv.imread(os.path.join(folderModePath,path)))


# Kameradan alınan götüntü okundu ve daha önceden belirlenen background üzerine eklenerek arayüz yapıldı.
while True:
    ret, frame = cap.read()
    img_background[162:162 + 480, 55:55 + 640] = frame
    img_background[44:44 + 633, 808:808 + 414] = imgModeList[0]

    cv.imshow('App', img_background)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break



