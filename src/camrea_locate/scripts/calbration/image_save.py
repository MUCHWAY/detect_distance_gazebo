#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2

capture=cv2.VideoCapture(0)
# capture.set
(6,cv2.VideoWriter.fourcc('M','J','P','G'))
# capture.set(3,1280)
# capture.set(4,720)
# capture.set(5,30)
print(capture.get(3),capture.get(4),capture.get(5))

name=0
while capture.isOpened():
    rec,img=capture.read()

    cv2.imshow("display",img)
    key=cv2.waitKey(5)
    print(key)

    if key==115:
        cv2.imwrite(str(name)+".jpg",img)
        name+=1
    



