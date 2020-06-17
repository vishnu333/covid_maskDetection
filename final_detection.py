from keras.models import load_model
import os
import cv2
import numpy as np


model = load_model('C:\\Users\\vishn\\holidays\\face_mask_detection\\models\\best_model.hdf5')

face_classifier = cv2.CascadeClassifier('C:\\Users\\vishn\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

src = cv2.VideoCapture(0)

lable = {0:'MASK',1:'WITHOUT MASK'}
color = {0:(0,255,0),1:(0,0,255)}

while True :

    ret,img = src.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img,1.3,5)

    for x,y,w,h in faces : 
        face_img = gray_img[y:y+w , x:x+w]
        resized = cv2.resize(face_img,(100,100))
        normalised = resized / 255.0
        reshaped = np.reshape(normalised,(1,100,100,1))
        result = model.predict(reshaped)

        lables = np.argmax(result,axis=1)[0]

        cv2.rectangle(img,(x,y),(x+w,y+h),color[lables],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color[lables],-1)
        cv2.putText(img,lable[lables],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)

    cv2.imshow('live',img)

    if cv2.waitKey(10)==ord('q'):
        break
