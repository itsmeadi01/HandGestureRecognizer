import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import matplotlib
from matplotlib import pyplot as plt

nb_classes = 5
img_rows, img_cols = 200, 200
img_channels = 1
A = np.zeros((96,96,1))

path2 = 'action_b'

def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist

def arraycreator(data):
    while i < 500:
        if(data[i]!=A and data[i+1]!=A):
           image[i] = data[i] + data[i+1]
    i = i + 2
    return(image)

def datax():
        #if __name__ == "__main__":
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten() for images in sorted(imlist)], dtype = 'f')
    print(immatrix.shape)
    label=np.ones((total_images,),dtype = int)
    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    print(label)
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
    (X, y) = (train_data[0],train_data[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # normalize
    X_train /= 255
    X_test /= 255
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    #print (Y_train)
    #print (len(X_train))
    #print(len(Y_train))
    return X_train, Y_train, X_test, Y_test, samples_per_class

if __name__ == '__main__':
    lable = modlistdir(path2)
    print(lable)

   


