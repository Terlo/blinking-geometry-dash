import cv2
import numpy as np
import dlib
import pyautogui
from zoom import audio_function,video_function
#print("running")
cv2.destroyAllWindows()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_SLIPPAGE=0.12  
RIGHT_SLIPPAGE=0.08       
# Define the position and font settings for the text
left_position_vertical = (50, 200)  # (x, y) coordinates
right_position_vertical = (50, 400)  # (x, y) coordinates
# Define the position and font settings for the text
left_position_horizontal = (150, 200)  # (x, y) coordinates
right_position_horizontal = (150, 400)  # (x, y) coordinates

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 0, 0)  # Text color (BGR format)
font_thickness = 2
 
def get_line_length(a,b):
   return(abs((b[0]-a[0])+(b[1]-a[1])))

def get_midpoint(a,b):
    return(int((a.x +b.x)/2),int((a.y+b.y)/2))

frame_count = 0
while True:
    _ , frame = cap.read()
    
    frame_count += 1
    
    # Check if it's an even-numbered frame (every second frame)
    if frame_count % 1 == 0:
        
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
            
            left_eye_horizontal = get_line_length(left_point_pf_the_eye,right_point_pf_the_eye)
            left_eye_vertical = get_line_length(mid_top,mid_bottom)
            # Use cv2.putText to add the text to the image
            #cv2.putText(frame, str(left_eye_horizontal), left_position_horizontal, font, font_scale, font_color, font_thickness)
            #cv2.putText(frame, str(left_eye_vertical), left_position_vertical, font, font_scale, font_color, font_thickness)

            #RIGHT EYE   
            left_point_pf_the_eye = (landmarks.part(42).x,landmarks.part(42).y)
            right_point_pf_the_eye = (landmarks.part(45).x,landmarks.part(45).y)
            cv2.line(frame,left_point_pf_the_eye,right_point_pf_the_eye,(0,255,0),2)

            #get the midpoint of the eye
            mid_top = get_midpoint(landmarks.part(43),landmarks.part(44))
            mid_bottom = get_midpoint(landmarks.part(47),landmarks.part(46))
            cv2.line(frame,mid_top,mid_bottom,(0,255,0),2)

            right_eye_horizontal = get_line_length(left_point_pf_the_eye,right_point_pf_the_eye)
            right_eye_vertical = get_line_length(mid_top,mid_bottom)
            # Use cv2.putText to add the text to the image
            #cv2.putText(frame, str(right_eye_horizontal), right_position_horizontal, font, font_scale, font_color, font_thickness)
            #cv2.putText(frame, str(right_eye_vertical), right_position_vertical, font, font_scale, font_color, font_thickness)

            left_eye_ratio  = left_eye_vertical/left_eye_horizontal
            right_eye_ratio  = right_eye_vertical/right_eye_horizontal

            left_eye_text = f"Left Eye Ratio: {left_eye_ratio:.2f}"
            right_eye_text = f"Right Eye Ratio: {right_eye_ratio:.2f}"
            cv2.putText(frame, left_eye_text, (50, 250), font, 0.75, (255,125,0), font_thickness)
            cv2.putText(frame, right_eye_text, (50, 450), font, 0.75, (255,125,0), font_thickness)
            # #print(f"L{left_eye_ratio} R{right_eye_ratio}")
            # left_eye_average.append(left_eye_ratio)
            # right_eye_average.append(right_eye_ratio)
            # ltotal  =sum(left_eye_average)
            # rtotal  =sum(right_eye_average)
            # laverage= ltotal/len(left_eye_average)
            # raverage= rtotal/len(right_eye_average)
            #print(f"LAVG:{laverage} RAVG:{raverage}")
            # Use cv2.putText to add the text to the image
            if left_eye_ratio< LEFT_SLIPPAGE or right_eye_ratio<RIGHT_SLIPPAGE:
                #print("left eye blink")
                cv2.putText(frame, "Jumping", (50, 50), font, font_scale, (255,255,0), font_thickness)
                pyautogui.press("space")
            #if right_eye_ratio< RIGHT_SLIPPAGE:
            #    #print("right eye blink")
            #    cv2.putText(frame, "Right Eye:Blink", (50, 375), font, font_scale, (0,0,255), font_thickness)
            #    pyautogui.press("space")    
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1)
        if key ==27:
            break  

cap.release()
cv2.destroyAllWindows()