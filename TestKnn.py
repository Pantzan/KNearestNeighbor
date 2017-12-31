from PythonKnn import PythonKnn
# Load the IRIS dataset, as in the labs
# %matplotlib inline


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#import k-nn classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import operator


iris = datasets.load_iris()

#view a description of the dataset (uncomment next line to do so)
#print(iris.DESCR)

#Set X equal to features, Y equal to the targets

X=iris.data 
y=iris.target 

mySeed=1234567

#initialize random seed generator 
np.random.seed(mySeed)

#we add some random noise to our data to make the task more challenging
X=X+np.random.normal(0,0.5,X.shape)

np.random.seed(mySeed)
indices= np.random.permutation(X.shape[0]) 
bins=np.array_split(indices,2) # we  just need a training and testing set here
foldTrain=bins[0]
foldTest=bins[1]
knn=PythonKnn(10,'euclidean')
knn.fit(X[foldTrain],y[foldTrain])
y_pred=knn.predict(X[foldTest])
a = np.where(y_pred != y[foldTest])
print(knn.accuracy(y[foldTest],y_pred))
# print(knn.confMat())
# print(knn.confMat(y[foldTest],y_pred,len(np.unique(y))))