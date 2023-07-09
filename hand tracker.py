import cv2
import mediapipe as mp
import time
from concurrent.futures import ThreadPoolExecutor

def hand():
    cap=cv2.VideoCapture(0)

    mpHands=mp.solutions.hands
    hands=mpHands.Hands(max_num_hands=6)
    mpDraw=mp.solutions.drawing_utils

    while True:
        succes,img=cap.read()
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img2=cv2.flip(img,1)
        results=hands.process(img2)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img2,handLms,mpHands.HAND_CONNECTIONS)

        cv2.imshow("Image",img2)
        cv2.waitKey(1)


        
with ThreadPoolExecutor() as executor:
    executor.submit(hand)
