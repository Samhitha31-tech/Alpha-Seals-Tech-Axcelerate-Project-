import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from math import hypot
bpoints=[deque(maxlen=1024)]
gpoints=[deque(maxlen=1024)]
rpoints=[deque(maxlen=1024)]
ypoints=[deque(maxlen=1024)]
colorIndex=0
eraser_mode=False
prev_point=None
drawing_enabled=False
colors=[(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
buttons=[(40,"CLEAR"),(140,"BLUE"),(240,"GREEN"),(340,"RED"),(440,"YELLOW"),(540,"ERASER")]
paintWindow=np.ones((471,636,3),np.uint8)*255
for i,(x,text) in enumerate(buttons):
    cv2.rectangle(paintWindow,(x,1),(x+80,51),(0,0,0) if text=="CLEAR" else colors[i-1] if i>0 and text!="ERASER" else (128,128,128),2)
    cv2.putText(paintWindow,text,(x +10,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
cv2.namedWindow("Paint",cv2.WINDOW_AUTOSIZE)
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)
    framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    for i,(x,text) in enumerate(buttons):
        cv2.rectangle(frame,(x,1),(x+80,51),(0,0,0) if text=="CLEAR" else colors[i-1] if i>0 and text!="ERASER" else(128,128,128),2)
        cv2.putText(frame,text,(x+10,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
    result=hands.process(framergb)
    if result.multi_hand_landmarks:
        landmarks=[]
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx,lmy=int(lm.x*640),int(lm.y*480)
                landmarks.append([lmx,lmy])
            mpDraw.draw_landmarks(frame,handslms,mpHands.HAND_CONNECTIONS)
        fore_finger=tuple(landmarks[8])
        middle_finger=tuple(landmarks[12])
        distance=hypot(fore_finger[0]-middle_finger[0],fore_finger[1]-middle_finger[1])
        drawing_enabled=distance>50
        if fore_finger[1]<=51:
            for i,(x,text) in enumerate(buttons):
                if x<=fore_finger[0]<=x+80:
                    if text=="CLEAR":
                        paintWindow[:]=255
                    elif text=="ERASER":
                        eraser_mode=True
                    else:
                        colorIndex = i-1
                        eraser_mode = False
                    break
        else:
            if drawing_enabled:
                if prev_point is not None:
                    cv2.line(frame,prev_point,fore_finger,(255,255,255) if eraser_mode else colors[colorIndex],5)
                    cv2.line(paintWindow,prev_point,fore_finger,(255,255,255) if eraser_mode else colors[colorIndex],5)
                prev_point=fore_finger
            else:
                prev_point=None
    else:
        prev_point = None
    cv2.imshow("Output",frame)
    cv2.imshow("Paint",paintWindow)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
