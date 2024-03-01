import cv2 as cv

# Kamera açma ve daha önceden tanımlanmış olan xml dosyasını okuma işlemlerini bu noktada gerçekleştiriyoruz.
cap = cv.VideoCapture(0)
haar_cacade = cv.CascadeClassifier('Xml Files\haar_face.xml')

while True:
    # Kameradan alınan görünütüyü okuma
    ret, fream =  cap.read()
    # Görseldeki gürültüyü azaltmak için görseli gri renkli bir yapıya çeviriyoruz
    gray = cv.cvtColor(fream, cv.COLOR_BGR2GRAY)
    # Haar cascade classifier ile yüzleri belirleyip bunları bulduğu kısımlara sınırlama koyuyoruz.
    faces_rect = haar_cacade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    # Bulunan her yüz için bir dikdörtgen oluşturuyoruz.
    for (x, y, w, h) in faces_rect:
        cv.rectangle(fream, (x, y),(x+w, y+h), (0,255,0),thickness=2)
    # Oluşturmuş olduğumuz bu dikdörtgenleri ekrana çıkartıyoruz
    cv.imshow('Detected Faces', fream)


   # 'q' tuşuna basılınca döngüden çık
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak
cap.release()

# Pencereleri kapat
cv.destroyAllWindows()