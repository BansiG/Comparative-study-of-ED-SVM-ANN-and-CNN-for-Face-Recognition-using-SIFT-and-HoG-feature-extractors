import cv2
import numpy as np 
import os
import random
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# path = "./face_data/"
# face_list = os.listdir(path)

# X_data = []
# labels = []
# for name in face_list:
# 	img = cv2.imread(path+name)
# 	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 	gray = cv2.resize(gray,(300,300))
# 	vect = gray.flatten()
# 	print vect.shape, name.split('.')[0].split('_')[0]
# 	X_data.append(vect)
# 	labels.append(name.split('.')[0].split('_')[0])


# print np.array(X_data).shape
# print np.array(labels).shape

# c = list(zip(np.array(X_data), labels))

# # np.save('face.npy',np.array(c))
# np.save('X.npy',np.array(X_data))
# np.save('Y.npy',np.array(labels))
# np.savetxt('X.csv',np.array(X_data))
# np.savetxt('Y.csv',np.array(labels))

print("Fetching data..")

# t = np.load('face.npy')

X_train= np.load('XTrain.npy')
y_train = np.load('YTrain.npy')
X_test= np.load('XTest.npy')
y_test = np.load('YTest.npy')


# c = list(zip(X,Y))
# random.shuffle(c)
# X , Y = zip(*c)
# print X,Y

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#print X_train.shape, y_train.shape
X_train = np.array(X_train)
y_train = np.array(y_train)
length = X_train.shape[0]
#print length
sift = cv2.xfeatures2d.SIFT_create()

descriptor = []
final_des = dict()
label = []
cnt = 0
new_des = []
for i in range(0,length):
	img_name= y_train[i]
	img_data = X_train[i]
	# print img_name
	img_data= np.reshape(img_data, (-1, 200))
	kp1, des1 = sift.detectAndCompute(img_data,None)
	# print des1.shape
	kmeans = KMeans(n_clusters=1, random_state=0).fit(des1)
	new_des.append(kmeans.cluster_centers_)
	label.append(img_name)
	# print (kmeans.cluster_centers_).shape
	final_des[img_name] = img_data
	cnt += des1.shape[0]



new_des = np.reshape(new_des,(length,128)) 	

#print np.array(new_des).shape
# print np.array(descriptor).shape
#print np.array(label).shape
# print len(final_des)
# print np.array(final_des).shape
#print "svm1"
c = list(zip(new_des,label))
random.shuffle(c)
new_des , label = zip(*c)

#print "svm"
clf = SVC(kernel = 'linear', C =100.0, gamma = 0.01)
#print "svm2"
clf.fit(new_des, label)


print("Testing on the image")

test_img = cv2.imread("F:\Desktop\ICT\AuthorPhoto_Bansi_Gajera.jpg",0)
test_img = cv2.resize(test_img,(200,200))
kp_test, des_test = sift.detectAndCompute(test_img,None)
kmeans_t = KMeans(n_clusters=1, random_state=0).fit(des_test)
#print np.array(kmeans_t.cluster_centers_).shape
print("Predicted roll number")



index= int(clf.predict(kmeans_t.cluster_centers_))
#print index
#print final_des[index]
cv2.imwrite("tt.jpg",final_des[index])
#print clf.score(new_des,label)


X_test = np.array(X_test)
length = X_test.shape[0]
final_des = dict()
label = []
cnt = 0
test_des = []
test_label = []
for i in range(0,length):
	img_name= y_test[i]
	img_data = X_test[i]
	# print img_name
	img_data= np.reshape(img_data, (-1, 200))
	kp1, des1 = sift.detectAndCompute(img_data,None)
	# print des1.shape
	kmeans = KMeans(n_clusters=1, random_state=0).fit(des1)
	test_des.append(kmeans.cluster_centers_)
	test_label.append(img_name)
	# print (kmeans.cluster_centers_).shape
	cnt += des1.shape[0]

	
precision = []
recall = []
	
j = 0
	
while(j<= len(X_train)):

	#for k in range(0,j):
	test_des = np.reshape(test_des[0:j],(j+1,128)) 	
#print(test_des.shape)
#print clf.score(test_des,test_label)
	y_score = clf.predict(test_des)
	
	f1score.append(f1_score(test_label, y_score, average="macro"))
	precision.append(precision_score(test_label, y_score, average="macro"))
	recall.append(recall_score(test_label, y_score, average="macro"))
	
	j = j+20
# test_label = np.reshape(test_label,(44))
# y_score = np.array(y_score)

# y_score = np.array(y_score)
# y_score = y_score.flatten()

# test_label = np.array(test_label)
# test_label = test_label.flatten()

# print y_score.shape , test_label.shape
# average_precision = average_precision_score(y_score, test_label)
# print('Average precision-recall score: {0:0.2f}'.format(
#       average_precision))