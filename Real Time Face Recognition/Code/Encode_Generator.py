import cv2 as cv
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-6060d-default-rtdb.firebaseio.com/",
    'storageBucket' : "faceattendacerealtime-6060d.appspot.com"
})

# Importing person images
folderImagesPath = 'Images'
imagesPathList = os.listdir(folderImagesPath)
imgList = []
personIdList = []
for path in imagesPathList:
    imgList.append(cv.imread(os.path.join(folderImagesPath,path)))
    personIdList.append(os.path.splitext(path)[0])

    fileName = f'{folderImagesPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

def findEncoding(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print('Encoding started ...')
encodeListKnown = findEncoding(imgList)
encodeListKnownIds = [encodeListKnown, personIdList]
print('Encoding complete')

file = open('EncodeFile.p','wb')
pickle.dump(encodeListKnownIds, file)
file.close()
print('File saved')