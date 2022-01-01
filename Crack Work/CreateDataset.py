# -*- coding: utf-8 -*-


import os 
import cv2
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
# Set up matplotlib fig, and size it to fit 4x4 pics
import matplotlib.image as mpimg
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split # Import train_test_split function
nrows = 4
ncols = 4

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index = 18000
train_Positive_fnames = os.listdir('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\Dataset\\Positive')      # path of original dataset
train_Negative_fnames = os.listdir('C:\\Users\HP\\Desktop\\IPCVProject2\\CrackWork\\Dataset\\Negative')	# path of original dataset
test_Positive_fnames = os.listdir('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\testing\\Positive') # path of test dataset floder where you want to save
test_Negative_fnames = os.listdir('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\testing\\Negative') # path of test dataset floder where you want to save
next_Positive_pix = list()
next_Negative_pix = list()
train_dir='C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\train'
test_dir='C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\test'
pos_train, pos_test, neg_train, neg_test = train_test_split(train_Positive_fnames, train_Negative_fnames, test_size=0.2, random_state=1) # 80% training and 20% test


for fn in pos_train:
    originalImage = cv2.imread('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\Dataset\\Positive'+"\\"+fn) # path of original dataset. don't remove fn
    cv2.imwrite('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\train\\Positive'+"\\"+fn,originalImage) # path of train dataset where you want to save. don't remove fn
for fn in pos_test:
    originalImage = cv2.imread('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\Dataset\\Positive'+"\\"+fn)
    cv2.imwrite('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\test\\Positive'+"\\"+fn,originalImage)
    
for fn in neg_train:
    originalImage = cv2.imread('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\Dataset\\Negative'+"\\"+fn)
    cv2.imwrite('train\\Negative'+"\\"+fn,originalImage)
    
for fn in neg_test:
    originalImage = cv2.imread('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\Dataset\\Negative'+"\\"+fn)
    cv2.imwrite('C:\\Users\\HP\\Desktop\\IPCVProject2\\CrackWork\\test\\Negative'+"\\"+fn,originalImage)

