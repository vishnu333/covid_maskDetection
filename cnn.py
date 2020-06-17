import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

data = np.load('C:\\Users\\vishn\\holidays\\face_mask_detection\\data.npy')
target = np.load('C:\\Users\\vishn\\holidays\\face_mask_detection\\target.npy')

model = Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.1)

model_checkpoint = ModelCheckpoint("C:\\Users\\vishn\\holidays\\face_mask_detection\\models\\best_model.hdf5",monitor='val_loss',verbose=1,save_best_only=True,mode='auto',period=1)

history = model.fit(train_data,train_target,epochs=20,validation_data=(test_data,test_target),callbacks=[model_checkpoint],validation_split=0.2)

model.save('C:\\Users\\vishn\\holidays\\face_mask_detection\\mainmodel.yml')
