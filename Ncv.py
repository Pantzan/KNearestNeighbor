from PythonKnn import PythonKnn

def NestedCrossVal(X,y,foldK,nns,dists,mySeed):

    np.random.seed(mySeed)
    accuracy_fold = []   
    indices = np.random.permutation(X.shape[0])
    bins = np.array_split(indices, foldK)
    sum_matrix = []
    assert(foldK == len(bins))

    for i in range(0,foldK):
        foldTrain=[] 
        foldTest=[]
        foldVal=[]
        # split the X to train, test and validation folds
        foldTest.extend(bins[i])

        if(i != foldK-1):
            foldVal.extend(bins[i+1])
        else:
            foldVal.extend(bins[0])  
        
        # don't add dublicates
        for j in range(0,len(bins)):
            if(i != j and i + 1 != j):
                if(i == foldK-1 and j == 0):                   
                    continue
                foldTrain.extend(bins[j])

        assert not np.intersect1d(foldTest,foldVal)
        assert not np.intersect1d(foldTrain,foldTest)
        assert not np.intersect1d(foldTrain,foldVal)
     
        bestDistance=''
        bestNN=-1 
        bestAccuracy=-10
        
        #loop to all available parametres and find the best to train the model
        for d in range(0, len(dists)):  
                 
            for k in nns:
                knn = PythonKnn(k, dists[d])
                knn.fit(X[foldTrain], y[foldTrain])              
                
                y_pred = knn.predict(X[foldVal])
                acc = knn.accuracy(y[foldVal], y_pred)
                #it prints all folds results. 
#                 print('-- validation test, N', k, 'Dist', dists[d],'accuracy ',acc )
                if acc > bestAccuracy:
                    bestAccuracy = acc
                    bestDistance = dists[d]
                    bestNN = k
                    
        print('** val for this fold, best NN', bestNN, 'best Dist', bestDistance)
        
        newTrain = foldTrain 
        newTrain.extend(foldVal)       
        
        # call the Knn and train with the best found parametres
        knn = PythonKnn(bestNN, metric=bestDistance)
        knn.fit(X[newTrain], y[newTrain])                
        y_pred = knn.predict(X[foldTest])
        acc = knn.accuracy(y[foldTest], y_pred)
        accuracy_fold.append(acc)      
        print('** End of test for this fold (',i+1,'), accuracy', acc)
        cm = knn.confMat(y[foldVal], y_pred, len(np.unique(y)))
        
        # append matrices to list for later use
        sum_matrix.append(cm)
        print(cm)
        
          # The sklearn KNN classifier for comparison purposes
#         from sklearn.neighbors import KNeighborsClassifier
#         knn2 = KNeighborsClassifier(n_neighbors=80, metric='euclidean')
#         knn2.fit(X[foldTrain], y[foldTrain])
#         y_pred2 = knn2.predict(X[foldTest])
#         acc2 = knn.accuracy(y[foldTest], y_pred)
#         print("2222", knn.accuracy(y[foldTest], y_pred2))
#         print(knn.confMat(y[foldVal], y_pred, len(np.unique(y))))

    print('==== Final Cross-val on test on this fold with NN', bestNN, 'dist', bestDistance, ' accuracy ',knn.accuracy(y[foldTest],y_pred))
    print('==== Final Confusion Matrix')
    
    #print the total confusion matrix
    print(sum(sum_matrix))
    return accuracy_fold
    

# print the results
accuracy_fold=NestedCrossVal(X,y,5,list(range(1,11)),['euclidean','absolute'],mySeed)
# print(np.mean(accuracy_fold))
print('==== Standard Deviation:', np.std(accuracy_fold))