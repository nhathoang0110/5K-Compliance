import pandas as pd
import random
import os
import shutil
random.seed(0)
import numpy as np
from sklearn.model_selection import StratifiedKFold


train=pd.read_csv('dataset/train/train_meta.csv')


img_list_0=[]
img_list_1=[]
for index, row in train.iterrows():
    if(row['mask']==1):
        img_list_1.append(row['fname'])
    if(row['mask']==0) :
        img_list_0.append(row['fname'])

img_list=img_list_0+img_list_1
label_list=[0]*len(img_list_0)+[1]*len(img_list_1)

img_list = np.array(img_list)
label_list = np.array(label_list)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
skf.get_n_splits(img_list, label_list)

fold = 1
for train_index, val_index in skf.split(img_list, label_list):
    X_train_fold = list(img_list[train_index])
    X_val_fold = list(img_list[val_index])

    y_train_fold = list(label_list[train_index])
    y_val_fold = list(label_list[val_index])

    os.makedirs("dataset/mask_kfolddata/fold"+str(fold)+"/train/0/")
    os.makedirs("dataset/mask_kfolddata/fold"+str(fold)+"/train/1/")
    os.makedirs("dataset/mask_kfolddata/fold"+str(fold)+"/val/0/")
    os.makedirs("dataset/mask_kfolddata/fold"+str(fold)+"/val/1/")

    for i in range(len(X_train_fold)):
        if(y_train_fold[i]==0):
            shutil.copy(os.path.join("dataset/train/images", X_train_fold[i]), "dataset/mask_kfolddata/fold"+str(fold)+"/train/0/")
        if(y_train_fold[i]==1):
            shutil.copy(os.path.join("dataset/train/images", X_train_fold[i]), "dataset/mask_kfolddata/fold"+str(fold)+"/train/1/")

    for i in range(len(X_val_fold)):
        if(y_val_fold[i]==0):
            shutil.copy(os.path.join("dataset/train/images", X_val_fold[i]), "dataset/mask_kfolddata/fold"+str(fold)+"/val/0/")
        if(y_val_fold[i]==1):
            shutil.copy(os.path.join("dataset/train/images", X_val_fold[i]), "dataset/mask_kfolddata/fold"+str(fold)+"/val/1/")

    f=open("dataset/mask_kfolddata/train_fold_"+str(fold)+".txt",'w')
    string=""
    for i in X_train_fold:
        string=string+i+"\n"
    f.write(string)
    f.close()

    f=open("dataset/mask_kfolddata/val_fold_"+str(fold)+".txt",'w')
    string=""
    for i in X_val_fold:
        string=string+i+"\n"
    f.write(string)
    f.close()
    fold += 1
