import cv2 as cv
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-6060d-default-rtdb.firebaseio.com/",
    'storageBucket' : "faceattendacerealtime-6060d.appspot.com"
})

bucket = storage.bucket()

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


modeType = 0
counter = 0
id = -1 
imgPerson = []

# Kameradan alınan götüntü okundu ve daha önceden belirlenen background üzerine eklenerek arayüz yapıldı.
while True:
    ret, frame = cap.read()

    imgS = cv.resize(frame, (0,0), None, 0.25, 0.25 )
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    img_background[162:162 + 480, 55:55 + 640] = frame
    img_background[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            img_background = cvzone.cornerRect(img_background, bbox, rt=0)
            id = personIdList[matchIndex]
            if counter == 0:
                counter = 1
                modeType = 1
    
    if counter != 0:
        if counter == 1:
            personInfo = db.reference(f'Persons/{id}').get()

            blob = bucket.get_blob(f'Images/{id}.jpg')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgPerson = cv.imdecode(array, cv.COLOR_BGRA2BGR)
        
        if 10<= counter <= 20:
            modeType = 2

        img_background[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if counter <= 10:
            cv.putText(img_background, str(personInfo['old']), (861, 125),
                    cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv.putText(img_background, str(personInfo['profession']), (1006, 550),
                    cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
            cv.putText(img_background, str(personInfo['id']), (1006, 493),
                    cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
            
            (w, h), _ = cv.getTextSize(personInfo['name'], cv.FONT_HERSHEY_COMPLEX,1, 1)
            offset = (414 -w)//2
            cv.putText(img_background, str(personInfo['name']), (808 + offset, 445),
                    cv.FONT_HERSHEY_COMPLEX, 1, (50,50,50), 1)
            
            imgPerson = cv.resize(imgPerson, (216, 216))
            img_background[175:175 + 216, 909:909 + 216] = imgPerson

        counter += 1

    cv.imshow('App', img_background)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break



