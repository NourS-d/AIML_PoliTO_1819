import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score,confusion_matrix

# Load Iris dataset
iris=datasets.load_iris()


# Select the first two dimensions
x=iris.data[:,:2]
y=iris.target
print('Dataset shape: '+ str(x.shape))

plt.scatter(x[:,0],x[:,1],c=y)
plt.savefig('./Output/1.png')
plt.show()

# Randomly split data into train, validation and test sets

X_tv, X_test, y_tv, y_test = train_test_split(
    x, y, test_size=0.3
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=2/7
)


# Training, Plotting, and Evaluation using linear SVM


x_min, x_max = X_train[:,0].min() - 1, X_train[:,0].max() + 1
y_min, y_max = X_train[:,1].min() - 1, X_train[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Cs= np.float_power(10,np.arange(-3,4,1))
acc=[]
for c in Cs:
    linear=svm.LinearSVC(C=c,max_iter=1000000000,multi_class='ovr')
    linear.fit(X_train,y_train)
    Z = linear.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
    plt.title("C=%f" %c)
    plt.savefig('./Output/'+str(c)+'.png')
    plt.show()

    y_valP=linear.predict(X_val)
    acc.append(accuracy_score(y_val,y_valP))

plt.plot(Cs,acc)
plt.xscale(value='log')
plt.savefig('./Output/Cs_acc.png')
plt.show()


#Test Set Evaluation
C=Cs[acc.index(np.max(acc))] #Returns lowest C with highest accuracy
linear = svm.LinearSVC(C=C, max_iter=1000000000, multi_class='ovr')
linear.fit(X_train, y_train)
y_testP=linear.predict(X_test)
print("The accuracy of the classifier is: ")
linScore=accuracy_score(y_test,y_testP)
print(linScore)


# RBF Kernel
acc=[]
for c in Cs:
    rbf=svm.SVC(C=c,kernel='rbf',gamma='auto')
    rbf.fit(X_train,y_train)
    Z = rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
    plt.title("C=%f" %c)
    plt.savefig('./Output/'+str(c)+'RBF.png')
    plt.show()

    y_valP=rbf.predict(X_val)
    acc.append(accuracy_score(y_val,y_valP))


# Evaluate best value of C
plt.plot(Cs,acc)
plt.xscale(value='log')
plt.savefig('./Output/RBF_CsAcc.png')

plt.show()

C=Cs[acc.index(np.max(acc))] #Returns lowest C with highest accuracy
rbf = svm.SVC(C=C,kernel='rbf',gamma='auto')
rbf.fit(X_train, y_train)
y_testP=rbf.predict(X_test)
print("The accuracy of the classifier is: ")
rbfScore=accuracy_score(y_test,y_testP)
print(accuracy_score(y_test,y_testP))


# Grid Search for both gamma and C

Cs= np.float_power(10,np.arange(-3,6,1))
Gs= np.float_power(10,np.arange(-9,3,1))
acc=np.zeros((len(Cs),len(Gs)))

for i,c in enumerate(Cs):
    for j,g in enumerate(Gs):
        rbf=svm.SVC(C=c,kernel='rbf',gamma=g)
        rbf.fit(X_train,y_train)
        y_valP=rbf.predict(X_val)
        acc[i,j]=accuracy_score(y_val,y_valP)

fig, ax = plt.subplots()
im = ax.imshow(acc)
ax.set_xticks(np.arange(len(Gs)))
ax.set_yticks(np.arange(len(Cs)))
ax.set_xticklabels(Gs)
ax.set_yticklabels(Cs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(Cs)):
    for j in range(len(Gs)):
        text = ax.text(j, i, "{0:0.1f}".format(acc[i,j]*100),
                       ha="center", va="center", color="w")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.savefig('./Output/gridSearch.png')
plt.show()

i,j=np.where(acc == np.max(acc))

Cf=i[0,]
Gf=j[0,]
rbf=svm.SVC(C=Cf,kernel='rbf',gamma=Gf)
rbf.fit(X_train,y_train)

#Evaluate on test set and decision boundary

y_testP=rbf.predict(X_test)
print("The accuracy of the classifier is: ")
rbfScoreImp=accuracy_score(y_test,y_testP)
print(rbfScoreImp)

Z = rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.title("C=%f, gamma=%f" %(Cf,Gf))
plt.savefig('./Output/bestRBF.png')
plt.show()

# The merged training and validation set is already available

# Grid search with CV

Cs= np.float_power(10,np.arange(-3,6,1))
Gs= np.float_power(10,np.arange(-9,3,1))
kf = KFold(n_splits=5)
kf.get_n_splits(X_tv,y_tv)
scores = np.empty((len(Cs), len(Gs), kf.n_splits))

for i, cvar in enumerate(Cs):
    for j, gvar in enumerate(Gs):
        for k,[train, test] in enumerate(kf.split(X_tv,y_tv)):
            rbf=svm.SVC(C=cvar,gamma=gvar)
            rbf.fit(X_tv[train], y_tv[train])
            scores[i,j,k] = rbf.score(X_tv[test],y_tv[test])


# This averages the value over each fold and plots the grid
average=np.zeros((len(Cs),len(Gs)))

for i in range(1,len(Cs)):
    for j in range(1,len(Gs)):
        average[i,j]=sum(scores[i,j,:])/kf.n_splits


fig, ax = plt.subplots()
im = ax.imshow(average)
ax.set_xticks(np.arange(len(Gs)))
ax.set_yticks(np.arange(len(Cs)))
ax.set_xticklabels(Gs)
ax.set_yticklabels(Cs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(Cs)):
    for j in range(len(Gs)):
        text = ax.text(j, i, "{0:0.1f}".format(average[i,j]*100),
                       ha="center", va="center", color="w")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.savefig('./Output/gridSearchNew.png')
plt.show()

# Max Accuracy
i,j=np.where(average == np.max(average))
Cf=i[0,]
Gf=j[0,]
rbf=svm.SVC(C=Cf,kernel='rbf',gamma=Gf)
rbf.fit(X_train,y_train)

#Evaluate on test set

y_testP=rbf.predict(X_test)
print("The accuracy of the classifier is: ")
rbfScoreImpG=accuracy_score(y_test,y_testP)
print(rbfScoreImpG)
