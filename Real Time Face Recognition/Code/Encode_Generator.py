import cv2 as cv
import face_recognition
import pickle
import os

# Importing person images
folderImagesPath = 'Images'
imagesPathList = os.listdir(folderImagesPath)
imgList = []
personIdList = []
for path in imagesPathList:
    imgList.append(cv.imread(os.path.join(folderImagesPath,path)))
    personIdList.append(os.path.splitext(path)[0])

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