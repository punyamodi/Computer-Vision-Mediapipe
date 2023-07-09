import cv2
import mediapipe as mp
import time
from concurrent.futures import ThreadPoolExecutor

def hello():
    cap=cv2.VideoCapture(0)

    mpFaceMesh=mp.solutions.face_mesh
    mpDraw=mp.solutions.drawing_utils
    FaceMesh=mpFaceMesh.FaceMesh(max_num_faces=6)
    drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    mpDraw=mp.solutions.drawing_utils

    mpHands=mp.solutions.hands
    hands=mpHands.Hands(max_num_hands=8)

    mpPose=mp.solutions.pose
    pose=mpPose.Pose()


    while True:
        succes,img=cap.read()
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img2=cv2.flip(img,1)
        results2=hands.process(img2)
        results3=pose.process(img2)
        results=FaceMesh.process(img2)              

        if results3.pose_landmarks:
            mpDraw.draw_landmarks(img2,results3.pose_landmarks,mpPose.POSE_CONNECTIONS)

        if results2.multi_hand_landmarks:
            for handLms in results2.multi_hand_landmarks:
                mpDraw.draw_landmarks(img2,handLms,mpHands.HAND_CONNECTIONS)
        
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img2,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
        
        cv2.imshow("Image",img2)
        cv2.waitKey(1)



## makes program go faster
with ThreadPoolExecutor() as executor:
    executor.submit(hello)
