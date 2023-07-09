import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

mpFaceMesh=mp.solutions.face_mesh
mpDraw=mp.solutions.drawing_utils
FaceMesh=mpFaceMesh.FaceMesh(max_num_faces=6,static_image_mode=True,min_detection_confidence=0.3)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    succes,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img2=cv2.flip(img,1)
    results=FaceMesh.process(img2)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img2,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)

    cv2.imshow("Image",img2)
    cv2.waitKey(1)
