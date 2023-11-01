import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)



detector = dlib.get_frontal_face_detector()

while True:
    _ , frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = detector (grey)

    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key ==27:
        break

cap.release()
