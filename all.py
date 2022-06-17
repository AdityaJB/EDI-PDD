from ast import Bytes
from io import BytesIO
from keras.models import load_model
import streamlit as st
import os
import cv2 as cv
import glob as gb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main(): 
    st.info(__doc__)
    file = st.file_uploader("Upload your Image",type = ["png","jpg"])
    show_file = st.empty()

    if not file:
        show_file.info("Please upload a file: {} ".format(' '.join(["png","jpg"])))
        return 
    if isinstance(file,BytesIO):
        show_file.image(file)
        model=load_model("model.hdf5")
        Size=224
        train_dir='E:/edi/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
        test_dir='E:/edi/test/test'
        train_generator=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,rescale=1/255.0,validation_split=0.1,).flow_from_directory(train_dir,batch_size=16,target_size=(Size,Size),subset="training",shuffle=True)
        classes=list(train_generator.class_indices.keys())
        X_test=[]
        for folder in os.listdir(test_dir):
               files=gb.glob(test_dir+'/*.JPG')
               for file in files :
                    img=cv.imread(file)
                    X_test.append(cv.resize(img,(Size,Size)))
        
        X_test=np.array(X_test)
        X_test=X_test/255.0
        valid_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0, validation_split=0.1).flow_from_directory(train_dir,batch_size=16,target_size=(Size,Size),subset='validation',shuffle=False)
        predictions = model.predict(valid_generator)
        y_pred=model.predict(X_test)
        predictions=y_pred.argmax(axis=1)
        fig=plt.figure(figsize=(6,6))
        plt.imshow(X_test[0])
        plt.title(classes[valid_generator.index_array[0]])
        st.pyplot(fig)
    # file.close()
main()