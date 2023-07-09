import cv2
import mediapipe as mp
import time 

mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

while True:
    succes,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img2=cv2.flip(img,1)
    results=pose.process(img2)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img2,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    
    cv2.imshow("Image",img2)
    cv2.waitKey(1)

with ThreadPoolExecutor() as executor:
    executor.submit(hello)
