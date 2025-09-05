from os import scandir, getcwd
import cv2
import numpy as np
import pandas as pd

def ls(ruta):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

width = 28
height = 28    # For high resolution images , change the width and height as 224 *224

"""Image to CSV and  Preprocessing using Area interpolation method_Test  images"""

test_abdomen_path = '/content/drive/MyDrive/Dataset/Spanish dataset/ABDOMEN/test_abdomen/'
test_thorax_path = '/content/drive/MyDrive/Dataset/Spanish dataset/THORAX/test_thorax/'
test_femur_path = '/content/drive/MyDrive/Dataset/Spanish dataset/FEMUR/test_femur/'
test_maternal_cervix_path ='/content/drive/MyDrive/Dataset/Spanish dataset/MATERAL CERVIX/test_cervix/'
test_other_path ='/content/drive/MyDrive/Dataset/Spanish dataset/OTHER/test_other/'
test_brain_path ='/content/drive/MyDrive/Dataset/Spanish dataset/BRAIN/test_brain/'
list_abdomen_test = []
list_thorax_test = []
list_femur_test = []
list_maternal_cervix_test = []
list_other_test = []
list_brain_test = []

files_0 = ls(test_abdomen_path)

for image in files_0:
	img = cv2.imread(test_abdomen_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),0)
	list_abdomen_test.append(imagen_entrada)
df_abdomen_test = pd.DataFrame(data=list_abdomen_test, index=[i for i in range(len(list_abdomen_test))])

files_1 = ls(test_thorax_path)
for image in files_1:
	img = cv2.imread(test_thorax_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),1)
	list_thorax_test.append(imagen_entrada)
df_thorax_test = pd.DataFrame(data=list_thorax_test, index=[i for i in range(len(list_thorax_test))])

files_2 = ls(test_femur_path)
for image in files_2:
	img = cv2.imread(test_femur_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),2)
	list_femur_test.append(imagen_entrada)
df_femur_test = pd.DataFrame(data=list_femur_test, index=[i for i in range(len(list_femur_test))])

files_3 = ls(test_maternal_cervix_path)
for image in files_3:
	img = cv2.imread(test_maternal_cervix_path+image,0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),3)
	list_maternal_cervix_test.append(imagen_entrada)
df_maternal_cervix_test = pd.DataFrame(data=list_maternal_cervix_test, index=[i for i in range(len(list_maternal_cervix_test))])

files_4 = ls(test_other_path)
for image in files_4:
	img = cv2.imread(test_other_path+image,0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),4)
	list_other_test.append(imagen_entrada)
df_other_test = pd.DataFrame(data=list_other_test, index=[i for i in range(len(list_other_test))])

files_5 = ls(test_brain_path)
for image in files_5:
	img = cv2.imread(test_brain_path+image,0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),5)
	list_brain_test.append(imagen_entrada)
df_brain_test = pd.DataFrame(data=list_brain_test, index=[i for i in range(len(list_brain_test))])

dframes_test = [df_abdomen_test, df_thorax_test, df_femur_test,df_maternal_cervix_test,df_other_test,df_brain_test]
df_test = pd.concat(dframes_test)
print(df_test)
file_path = '/content/drive/MyDrive/Dataset/npy files/test_spanish.csv'
df_test.to_csv(file_path, header=None, index=False)

"""Image to CSV and  Preprocessing using Area interpolation method_Train  images"""

train_abdomen_path = '/content/drive/MyDrive/Dataset/Spanish dataset/ABDOMEN/training_abdomen/'
train_thorax_path = '/content/drive/MyDrive/Dataset/Spanish dataset/THORAX/training1_thorax/'
train_femur_path = '/content/drive/MyDrive/Dataset/Spanish dataset/FEMUR/training1_femur/'
train_maternal_cervix_path ='/content/drive/MyDrive/Dataset/Spanish dataset/MATERAL CERVIX/training_cervix/'
train_other_path ='/content/drive/MyDrive/Dataset/Spanish dataset/OTHER/training_other/'
train_brain_path ='/content/drive/MyDrive/Dataset/Spanish dataset/BRAIN/training_brain/'
list_abdomen_train = []
list_thorax_train = []
list_femur_train = []
list_maternal_cervix_train = []
list_other_train = []
list_brain_train = []

files_0 = ls(train_abdomen_path)

for image in files_0:
	img = cv2.imread(train_abdomen_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),0)
	list_abdomen_train.append(imagen_entrada)

df_abdomen_train = pd.DataFrame(data=list_abdomen_train, index=[i for i in range(len(list_abdomen_train))])

files_1 = ls(train_thorax_path)
print(len(files_1))
for image in files_1:
	img = cv2.imread(train_thorax_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),1)
	list_thorax_train.append(imagen_entrada)

df_thorax_train = pd.DataFrame(data=list_thorax_train, index=[i for i in range(len(list_thorax_train))])

files_2 = ls(train_femur_path)
for image in files_2:
	img = cv2.imread(train_femur_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),2)
	list_femur_train.append(imagen_entrada)

df_femur_train = pd.DataFrame(data=list_femur_train, index=[i for i in range(len(list_femur_train))])

files_3 = ls(train_maternal_cervix_path)
for image in files_3:
	img = cv2.imread(train_maternal_cervix_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),3)
	list_maternal_cervix_train.append(imagen_entrada)

df_maternal_cervix_train = pd.DataFrame(data=list_maternal_cervix_train, index=[i for i in range(len(list_maternal_cervix_train))])

files_4 = ls(train_other_path)
for image in files_4:
	img = cv2.imread(train_other_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),4)
	list_other_train.append(imagen_entrada)

df_other_train = pd.DataFrame(data=list_other_train, index=[i for i in range(len(list_other_train))])

files_5 = ls(train_brain_path)
for image in files_5:
	img = cv2.imread(train_brain_path+image, 0)
	dim = (width, height)
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),5)
	list_brain_train.append(imagen_entrada)

df_brain_train = pd.DataFrame(data=list_brain_train, index=[i for i in range(len(list_brain_train))])

dframes_train = [df_abdomen_train, df_thorax_train, df_femur_train,df_maternal_cervix_train,df_other_train,df_brain_train]
df_train = pd.concat(dframes_train)
print(df_train)
file_path = '/content/drive/MyDrive/Dataset/npy files/train_fetus_spanish.csv'
df_train.to_csv(file_path, header=None, index=False)
