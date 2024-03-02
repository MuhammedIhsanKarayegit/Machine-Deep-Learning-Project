import cv2 as cv
import os
import pickle
import face_recognition
import numpy as np
import cvzone


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


# Load encoding file
print('Loading Encode File ...')
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, personIdList = encodeListKnownWithIds
print('Encode File Loaded.')


# Kameradan alınan götüntü okundu ve daha önceden belirlenen background üzerine eklenerek arayüz yapıldı.
while True:
    ret, frame = cap.read()

    imgS = cv.resize(frame, (0,0), None, 0.25, 0.25 )
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    img_background[162:162 + 480, 55:55 + 640] = frame
    img_background[44:44 + 633, 808:808 + 414] = imgModeList[0]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            img_background = cvzone.cornerRect(img_background, bbox, rt=0)

    cv.imshow('App', img_background)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break



