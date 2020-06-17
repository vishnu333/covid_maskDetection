import os
import cv2
import numpy as np
import sys
from keras.utils import np_utils 


data_path='C:\\Users\\vishn\\holidays\\face_mask_detection\\dataset'
categories = os.listdir(data_path)
# print(categories)
labels = [i for i in range(len(categories))]
# print(labels)
label_dict = dict(zip(categories,labels))
# print(label_dict)
img_size = 100
data = []
target = []
for category in categories :
    folder_path = os.path.join(data_path,category)
    img_names = os.listdir(folder_path) 
    for image_name in img_names :
        img_path = os.path.join(folder_path,image_name)
        img = cv2.imread(img_path)
        try:    
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img,(img_size,img_size))
            data.append(resized_img)
            target.append(label_dict[category])
        except expression as identifier:
            print('exception : ',identifier)
data = np.array(data)/255.0
# print(data)
data = np.reshape(data,(data.shape[0],img_size,img_size,1))
target = np.array(target)
# print(data.shape)
# print(data)

target = np_utils.to_categorical(target)

np.save('C:\\Users\\vishn\\holidays\\face_mask_detection\\data',data)
np.save('C:\\Users\\vishn\\holidays\\face_mask_detection\\target',target)