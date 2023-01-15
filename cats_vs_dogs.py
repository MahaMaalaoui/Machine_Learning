#DataSet can be found on https://www.kaggle.com/datasets/salader/dogs-vs-cats

####################
#importing pakages
####################

import pandas as pd
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

####################
#viewdata
####################

image=Image.open('D:/workspace/ML/data_cat_dog/cat.170.jpg')
image=np.array(image)

plt.imshow(image)
plt.colorbar()

image.shape

############################################################
# function to read images from a specifique directory
############################################################

def get_data_set(filepath):
    imgs=[]
    labels=[]
    for f in os.listdir(filepath):
            labels.append(f.split('.')[0])
            img=np.resize((Image.open(filepath+f)),(224,224,3))
            imgs.append(np.asarray(img))
    
    return np.asarray(imgs),labels


####################
#importing data
####################

filepath = 'D:/workspace/ML/data_cat_dog/'
imgs,labels= get_data_set(filepath)  

####################
#Data augmentation
####################

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical"), tf.keras.layers.RandomRotation(0.2)])
images=[]
for y in imgs:
     images.append(y)
     for i in range(9):
        augmented_image = data_augmentation(y)
        images.append(augmented_image)

images=np.asarray(images)

####################
#label augmentation
####################

lab=[]
for x in labels:
 lab.append([x]*10)     

lab=np.reshape(lab,-1)


########################################
#transforming label into Numerique
#########################################

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder().fit_transform(lab)


####################
#shuffling data
####################

from sklearn.utils import shuffle
x, y = shuffle(images, lab, random_state=0)

####################
#test train split
####################

from sklearn.model_selection import train_test_split
x_train,x_test,train_label,test_label=train_test_split(x,y,test_size=0.33,random_state=0) 

#####################
#building CNN model
#####################

from keras.models import Sequential
classifier= Sequential()
classifier.add(Convolution2D(32,kernel_size=3,input_shape=(224,224,3),activation ='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile('adam',loss='binary_crossentropy',metrics=['accuracy'])



from tensorflow.keras.utils import plot_model
plot_model(classifier)

classifier.summary()


#####################
#training data
#####################
history=classifier.fit(x_train,train_label,batch_size=128,epochs=15,validation_data=(x_test,test_label))

###########################
#ploting accuracy and loss
###########################

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot( acc, 'b', label='Training Accuracy')
plt.plot( val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, 'b', label='Training Loss')
plt.plot(val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()



##########################################
#building VGG16 model using transfer learning
##########################################

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model


res = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
outputs = res.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation="sigmoid")(outputs)
clas_VGG = Model(inputs=res.input, outputs=outputs)
for layer in res.layers:
  layer.trainable = False
clas_VGG.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



hist_VGG=clas_VGG.fit(x_train,train_label,batch_size=128,epochs=20,validation_data=(x_test,test_label))




