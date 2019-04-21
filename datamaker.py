import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import dataextract as dx
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
i=1
while True:
    _, frame = cap.read()
    cv2.rectangle(frame,(300,250),(500,450),(0,255,0),1)
    frame1=frame[280:500,300:500]
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 52])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    cv2.imshow('mask', mask)
    cv2.imshow('frame',frame)
    if i==1:
        if cv2.waitKey(1) & 0xFF ==ord('r'):
            cv2.imwrite('iiok'+ str(i) +'.jpg',mask)
            i=i+1
    if i>1:
        cv2.imwrite("iiok"+ str(i) + ".jpg",mask)
        i=i+1

    if i==800:
        break

cv2.destroyAllWindows()
cap.release()
