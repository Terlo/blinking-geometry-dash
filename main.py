import cv2
import numpy as np
import dlib

print("running")
cv2.destroyAllWindows()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_midpoint(a,b):
    return(int((a.x +b.x)/2),int((a.y+b.y)/2))

while True:
    _ , frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = detector (grey)

    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)
        landmarks = predictor(grey, face)
        
        #LEFT EYE
        left_point_pf_the_eye = (landmarks.part(36).x,landmarks.part(36).y)
        right_point_pf_the_eye = (landmarks.part(39).x,landmarks.part(39).y)
        cv2.line(frame,left_point_pf_the_eye,right_point_pf_the_eye,(0,255,0),2)

        #get the midpoint of the eye
        mid_top = get_midpoint(landmarks.part(37),landmarks.part(38))
        mid_bottom = get_midpoint(landmarks.part(40),landmarks.part(41))
        cv2.line(frame,mid_top,mid_bottom,(0,255,0),2)
        
        #RIGHT EYE
        left_point_pf_the_eye = (landmarks.part(42).x,landmarks.part(42).y)
        right_point_pf_the_eye = (landmarks.part(45).x,landmarks.part(45).y)
        cv2.line(frame,left_point_pf_the_eye,right_point_pf_the_eye,(0,255,0),2)

        #get the midpoint of the eye
        mid_top = get_midpoint(landmarks.part(43),landmarks.part(44))
        mid_bottom = get_midpoint(landmarks.part(47),landmarks.part(46))
        cv2.line(frame,mid_top,mid_bottom,(0,255,0),2)





    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key ==27:
        break

cap.release()
cv2.destroyAllWindows()