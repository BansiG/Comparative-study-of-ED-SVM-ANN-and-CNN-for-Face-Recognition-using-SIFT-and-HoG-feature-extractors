import numpy as np 
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)

def data_preprocess(y,x): 
    
    y = map(int,y)
    un_label = []
    y_t = []
    for i in y:	
        if i not in un_label:
            un_label.append(i)
        temp = np.zeros(31,dtype=int)
        np.put(temp,un_label.index(i),1)
        y_t.append(temp)
    X = np.array(x)
    y = np.array(y_t)
    print (un_label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # np.save('X_train',X_train)
    # np.save('X_test', X_test)
    # np.save('y_train', y_train)
    # np.save('y_test', y_test)
    
    return un_label,y,X_train, X_test, y_train, y_test

def Train_data(dataset_file,img_data,img_name):
	t = np.load(dataset_file)
	length = t.shape[0]
	cnt = 0
	for i in range(0,length):
		img_name.append(t[i][1].split('_')[0])
	#	print img_name
		img_data.append(t[i][0])
	return img_data,img_name


def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1,300, 300), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(optimizer = 'adam', loss = 'mean_squared_error' , metrics = ['accuracy'])
	return model

img_data = []
img_name = []	
train_descriptor=Train_data("face.npy",img_data,img_name)

img_data = np.array(img_data)
img_name = np.array(img_name)

#print img_data.shape , img_name.shape
un_label,y_coded,X_train, X_test, y_train, y_test = data_preprocess(img_name,img_data)
#print np.array(un_label).shape

#print X_train.shape, X_test.shape
X_train = X_train.reshape(X_train.shape[0],1,-1,300).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1,-1, 300).astype('float32')
#print X_train.shape , X_test.shape , y_test.shape

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

num_classes = y_test.shape[1]



#print y_coded.shape

model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))