import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection()

while True:
    succes,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img2=cv2.flip(img,1)
    results=faceDetection.process(img2)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img2,detection)
    
    cv2.imshow("Image",img2)
    cv2.waitKey(1)
