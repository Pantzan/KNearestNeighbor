import numpy as np
class Knn:
    # initialize the object and it's variables
    def __init__(self, n, metric):
        self.predicted = []
        self.X = None
        self.y = None
        self.metric = metric
        self.n = n
    
    #Fit function to get X and y from IRIS dataset.
    def fit(self, Xx_, yy_):
        if(len(Xx_) != len(yy_)):
            raise ValueError("Data not equal in size with labels")
        self.X = Xx_
        self.y = yy_
    
    # Get the accuracy of the model and raise errors if the array sizes are wrong
    def accuracy(self, y_test,y_pred):
        if(len(y_test) != len(y_pred)):
            raise ValueError("Prediction and labels are not equal in size")
        a = np.where(y_test == y_pred)
        accuracy =  (float(len(a[0])) / len(y_test))
        return accuracy
    
    # Get the euclidean distance from two arrays by ziping them with Pythonic way
    def euclideanDistance(self, in1,in2):
        import math
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(in1, in2)]))
    
    # Get the absolute distance from two arrays for experiment purposes
    def absoluteDistance(self, in1, in2):
        return float(abs(sum([(a - b) for a, b in zip(in1, in2)])))

    # check the distance from all points
    def getNeighbours(self, x_):
        dists = []
        X = self.X
        metric = self.metric
        
        for i in range(0, X.shape[0]-1):
            if metric == 'euclidean':
                e = (( i, self.euclideanDistance(X[i], x_) ))
            elif metric == 'absolute':
                e = (( i, self.absoluteDistance(X[i], x_) ))
            else:
                raise ValueError("I don't understand this metric. Please choose 'euclidean' or 'absolute'" )
#             print(X[i], x_, e[1])
            dists.append(e)
    
        # sort the tuple by the distance index
        import operator
        s = dists.copy()
        s.sort(key=operator.itemgetter(1))
        # get the n neighbours
        s = s[:self.n]   
        # get their labels
        label = self.getLabel(s)
        
        # append to the array the calculated label
        self.assignLabel(label)
        return label 

    # get the labels from n neighbours
    def getLabel(self, neighbours):
        from collections import Counter
        indexes = []
        a_labels = []
        labels = []
        
        # check the distance from the first item in the array if neighbours.
        #If the distance is zero, then means  we found jackpot and returning this label as 100% correct prediction
        first = neighbours[0]
        if (first[1]) == 0:
            label = (self.y[first[0]],0)
        # get the majority of the labels and return it as the calculated label   
        else:        
            for i in range(0, len(neighbours)):
                index = neighbours[i][0]    
                a_labels.append(self.y[index])    

            label = Counter(a_labels).most_common(1)[0]
       
        # return only the label
        return label[0]    
    
    # append to predicted array the correct label for each index of x test
    def assignLabel(self, label):
        self.predicted.append(label)
        return self.predicted 

    # the main function. Calls all the helper Knn functions and returns the predicted array
    # It also does checks about empty or not identical in size arrays
    def predict(self, x_): 
        if (len(x_) > len(self.X)):
            raise ValueError("Test array is larger than the training data")
        elif (len(x_) == 0):
            raise ValueError("Test array is empty")
        for i in x_: 
            self.getNeighbours(i)
        return self.predicted
    
    # calculates the confusion matrix according to the prediction.
    def confMat(self, y_test_,y_pred_,classno):
        matrix = np.zeros(9).reshape(classno,classno)
        for i in range(0,y_test_.size):
            matrix[y_test_[i], y_pred_[i]] += 1    
        return matrix