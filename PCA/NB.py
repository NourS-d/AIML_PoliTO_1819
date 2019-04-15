from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
data=[]
y=[]

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

X_t=data_standardized

# Data Split
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y, test_size=0.1,random_state=6)

np.save('./Data/xtest.npy',X_test)
np.save('./Data/ytest.npy',y_test)
np.save('./Data/xtrain.npy',X_train)
np.save('./Data/ytrain.npy',y_train)


# Naive Bayes
weights=[1/4, 1/4, 1/4, 1/4]
gnb=GaussianNB(priors=weights)
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
print("Our accuracy without dimension reduction is: ",end="")
print(accuracy_score(y_test,y_pred))
print("The confusion matrix is:")
print(confusion_matrix(y_test,y_pred))

# Reduce Dimensionality 1st and 2nd


pca=PCA(4)
X_t=pca.fit_transform(data_standardized)
comp=pca.components_[0:2]
X_t=np.dot(data_standardized, comp.transpose())

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y, test_size=0.1,random_state=6)
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
print("Our accuracy with dimension reduction using 1st and 2nd PCs is: ",end="")
print(accuracy_score(y_test,y_pred))
print("The confusion matrix is:")
print(confusion_matrix(y_test,y_pred))


x_min, x_max = X_t[:,0].min() - 1, X_t[:,0].max() + 1
y_min, y_max = X_t[:,1].min() - 1, X_t[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))


Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#['dog':'purple','guitar':'blue', 'house':'green','person':'yellow']
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_t[:,0],X_t[:,1],c=y)
plt.savefig('./Run/Descision.png')
plt.show()


#Reduce dimensionality 3rd and 4th

comp=pca.components_[2:]
X_t=np.dot(data_standardized, comp.transpose())

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y, test_size=0.1,random_state=6)
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
print("Our accuracy with dimension reduction using 3rd and 4th PCs is: ",end="")
print(accuracy_score(y_test,y_pred))
print("The confusion matrix is:")
print(confusion_matrix(y_test,y_pred))
