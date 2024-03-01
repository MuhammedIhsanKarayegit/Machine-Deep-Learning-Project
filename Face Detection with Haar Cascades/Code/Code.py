import cv2 as cv

cap = cv.VideoCapture(0)
haar_cacade = cv.CascadeClassifier('Xml Files\haar_face.xml')

while True:

    ret, fream =  cap.read()

    gray = cv.cvtColor(fream, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cacade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(fream, (x, y),(x+w, y+h), (0,255,0),thickness=2)
    
    cv.imshow('Detected Faces', fream)


   # 'q' tuşuna basılınca döngüden çık
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak
cap.release()

# Pencereleri kapat
cv.destroyAllWindows()