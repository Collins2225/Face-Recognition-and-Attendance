import cv2
import numpy as np
import face_recognition
from altair import Facet
from face_recognition import face_distance

# Load the reference image
imgElon = face_recognition.load_image_file('ImagesBasic/Elon musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Load the test image
imgTest = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect face in the REFERENCE image (imgElon)
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

# Detect face in the TEST image (imgTest)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeElonTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 255, 0), 2)

results = face_recognition.compare_faces([encodeElon],encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon], encodeElonTest)
print(results,faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0])}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
