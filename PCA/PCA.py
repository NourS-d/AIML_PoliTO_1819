from PIL import Image
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


def PrincipleComponent(compNum, data):
    pca = PCA(compNum)
    X_t = pca.fit_transform(data)
    approximation = pca.inverse_transform(X_t)
    return X_t,approximation

# All images are stored in the Images Folder

data=[] # data
y=[]    # Label


# Load Images
directories=os.listdir('./Images/')
print("Start Loading")
for j in directories:
    allfiles = os.listdir('./Images/' + j + '/')
    imlist = [filename for filename in allfiles if filename[-4:] in ['.jpg']]
    for i in range(len(imlist)):
        img_data=np.asarray(Image.open('./Images/' + j + '/' + imlist[i]))

        # Data matrix update
        data.append(img_data.ravel())

        # Label vector update
        if (j == 'dog'):
            y.append(1)
        if(j=='guitar'):
            y.append(2)
        if(j=='house'):
            y.append(3)
        if(j=='person'):
            y.append(4)

shape=img_data.shape
print("Done loading images")


# Standardize Data
print("Standardizing the data")
scaler = preprocessing.StandardScaler()
data_standardized= scaler.fit_transform(data)


# PCA
imgNum=random.randint(0,len(data)-1)
imgNum=180
print("We will visualize image " + str(imgNum))

imgOrig = np.asarray(data[imgNum]).reshape(shape)

# 2 Components
X_t,approx=PrincipleComponent(2,data_standardized)
im2=np.asarray(approx[imgNum])
im2=scaler.inverse_transform(im2)
im2 = im2.reshape(shape)/255

# 6 Components
X_t,approx=PrincipleComponent(6,data_standardized)
im6=np.asarray(approx[imgNum])
im6=scaler.inverse_transform(im6)
im6 = im6.reshape(shape)/255

# 60 Components
X_t,approx=PrincipleComponent(60,data_standardized)
im60=np.asarray(approx[imgNum])
im60=scaler.inverse_transform(im60)
im60 = im60.reshape(shape)/255

# Last 6 PC
pca=PCA()
X_t=pca.fit_transform(data_standardized)
comp=pca.components_[-6:]
X_tL6=np.dot(data_standardized, comp.transpose())
data6=np.dot(X_tL6,comp)

imL6=np.asarray(data6[imgNum])
imL6=scaler.inverse_transform(imL6)
imL6 = imL6.reshape(shape)/255

# Image Plots


# Original image
plt.imshow(imgOrig)
plt.title('Original Image', fontsize = 20)
plt.savefig('./Run/orig.png')
plt.show()

 # 2 principal components
plt.subplot(2, 2, 1)
plt.imshow(im2)
plt.axis('off')
plt.title('2 PC', fontsize = 20)

 # 6 principal components
plt.subplot(2, 2, 2)
plt.imshow(im6)
plt.axis('off')
plt.title('6 PC', fontsize = 20)

 # 60 principal components
plt.subplot(2, 2, 3)
plt.imshow(im60)
plt.axis('off')
plt.title('60 PC', fontsize = 20)

# Last 6 principal components
plt.subplot(2, 2, 4)
plt.imshow(imL6)
plt.axis('off')
plt.title('Last 6 PC', fontsize=20)

plt.savefig('./Run/PCA.png')
plt.show()

# Visualization of data

color=['r','y','g','b']

for j in [0,2,9]:
    for i in range(len(y)):
        plt.scatter(X_t[i,j],X_t[i,j+1],c=color[y[i]-1])
    plt.title('%i and %i PC' %(j+1,j+2), fontsize = 20)
    plt.savefig('./Run/%i.jpg' %j)
    plt.show()



# Mean of our data
mean=scaler.mean_/255
plt.imshow(mean.reshape(shape))
plt.title('Data Mean', fontsize=20)
plt.savefig('./Run/mean.png')
plt.show()


# Explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.savefig('./Run/exVar.png')
plt.show()

