import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import dataextract as dx
import cv2
import numpy as np
convnet=input_data(shape=[None,200,200,1],name='input')

convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)

convnet=fully_connected(convnet,5,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')

model=tflearn.DNN(convnet,tensorboard_verbose=0)
model.load("GestureRecogModel.tfl")
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.rectangle(frame,(280,250),(500,450),(0,255,0),1)
    frame1=frame[300:500,300:500]
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 52])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    cv2.imshow('mask', mask)
    cv2.imshow('frame',frame)
    mask= mask.reshape(-1,200,200,1)
    print(model.predict(mask))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
