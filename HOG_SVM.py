
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog as hg
from skimage import data, exposure
import cv2
import os
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split


# In[60]:


Path2 = 'F:/Desktop/ICT/sem_6/ML/train/new_faces/'


# files2 = os.listdir(Path2)
# images = []


# for name in files2:
#     print(name)
    
#     temp = cv2.imread(Path2 + name)
# #     cv2.imwrite('F:/Desktop/extra.jpg',temp)
# #     cv2.imshow('F:/Desktop/extra.jpg')
# #     cv2.waitKey(0)
# #     temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
#     temp = cv2.resize(temp, (300,300), interpolation = cv2.INTER_AREA)
#     images.append(temp.flatten())

tempimage = []
        
data = []
label = []
unqiue_label = []
for files in os.listdir('F:/Desktop/ICT/sem_6/ML/Ass2/Dataset'):
    f = files.split('_')[0].split('O')[0].split('o')[0]
    if(f!='.DS'):
        tempimage.append(cv2.imread('F:/Desktop/ICT/sem_6/ML/Ass2/Dataset/'+files , 0))
#         tempgray = cv2.cvtColor(tempimage, cv2.COLOR_BGR2GRAY)  
#         tempresize = cv2.resize(tempgray,(100,100),interpolation = cv2.INTER_AREA)   
#         tempNP = np.array(tempresize)
#         tempNP = tempNP.flatten()

        

#         data.append(fd);
        label.append(f);
        if( f not in unqiue_label ):
            print(f)
            unqiue_label.append(f)

classes = len(unqiue_label)
np.save('x.npy',tempimage)
np.save('y.npy', label)
X_train, X_test, y_train, y_test = train_test_split( tempimage, label, test_size=0.5)

# X_train = np.load('F:/Desktop/ICT/sem 7/CV/Assignment1/XTrain.npy')

# y_train = np.load('F:/Desktop/ICT/sem 7/CV/Assignment1/YTrain.npy')

# X_test = np.load('F:/Desktop/ICT/sem 7/CV/Assignment1/XTest.npy')

# y_test = np.load('F:/Desktop/ICT/sem 7/CV/Assignment1/YTest.npy')

# for image in X_train:
            
# #     print(image.shape)
#     fd, hog_image = hg(image, orientations=9, pixels_per_cell=(8, 8),
#                         cells_per_block=(2, 2), visualise= True)
#     data.append(fd)
        
# data = np.array(data)
# data = data
# print("data")
# print(data.shape)
# print(np.array(label).shape)


# In[ ]:


print(len(unqiue_label))


# In[21]:


svm = SVC(C=50.0,kernel='linear',gamma=0.01,probability=True,verbose=2)
svm.fit(data,y_train)


# In[44]:


from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import precision_recall_fscore_support as score


cnt=0

precision = []
recall = []
accuracy = []

j = 20
while(j<= len(X_test) and j>=0):
    y_score = []
    for i in range(0,j):

        fd, hog_image = hg(X_test[i], orientations=10, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualise= True)



    #     svm.predict(np.reshape(fd,(1,-1)))
        if(y_test[i] == svm.predict(np.reshape(fd,(1,-1)))):

            cnt+=1
        y_score.append(svm.predict(np.reshape(fd,(1,-1))))
        
    
#     precision = (accuracy_score(y_test, y_score))
    test = y_test[0:j]
    print(len(test))
    print(len(y_score))
    precision.append(precision_score(test, y_score, average = 'macro'))

    recall.append(recall_score(test, y_score, average = 'macro'))
    
    accuracy.append(accuracy_score(y_test, y_score), average = 'macro')
    j = j+20
# print(' F1 score: {0:0.4f}'.format(f1_score(y_test, y_score, average = 'macro')))
    
    
# print(cnt, len(X_test))
    
 


# In[52]:


# print((precision))
# print(recall)
plt.plot(precision)
plt.plot(recall)
plt.plot(accuracy)
plt.title('HOG + SVM')
plt.xlabel('no. of test images ( x20)')


# In[19]:


from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import precision_recall_fscore_support as score

# precision, recall, fscore, support = score(y_test, y_score)

# # print(precision, recall, fscore, support)

# print('precision: '.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))

# print(len(y_test), len(y_score))
# average_accuracy = accuracy_score(y_test, y_score)
# for i in range(classes):
#     precision[i], recall[i], _ = precision_recall_curve(y_test[:,i],y_score[:,i])
#     plt.figure()
# plt.step(recall, precision['micro'], color='b', alpha=0.2,
#          where='post')

print(' Accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_score)))

print(' precision score: {0:0.4f}'.format(precision_score(y_test, y_score, average = 'macro')))

print(' Recall score: {0:0.4f}'.format(recall_score(y_test, y_score, average = 'macro')))

print(' F1 score: {0:0.4f}'.format(f1_score(y_test, y_score, average = 'macro')))


# In[22]:



# fd = {}


# for image in images:

image = cv2.imread('F:/Desktop/ICT/sem_6/ML/train/new_faces/' + '201501028_Anger.jpg',0)

fd, hog_image = hg(image, orientations=10, pixels_per_cell=(16, 16),
                cells_per_block=(2,2), visualise= True)
print(hog_image.shape)
print(fd.shape)
plt.imshow(hog_image)
# plt.imshow(fd)


# In[4]:


from skimage import exposure

(H, hogImage) = hg(X_train[0], orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), 
                   transform_sqrt=True, block_norm="L1", visualise=True)

hogImage = exposure.rescale_intensity(hogImage, in_range=(0, 0.05))
plt.hist(hogImage)
# hogImage = hogImage.astype("uint8")
# plt.imshow(hogImage)

