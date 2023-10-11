
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

# # Exercise 7: Support Vector Machine (SVM) 
# Perform classification for the entire MNIST dataset by using SVMs, e.g. functions in Scikit-learn or Matlab. 
# 

# ## Load data training and testing data
def create_complete_datasets(data_dict):
    '''
    Function for creating complete training and test sets containing
    all classes.
    '''
    #Empty list
    trainset = []
    traintargets =[]
    testset = []
    testtargets =[]
    
    #For each class
    for i in range(10):
        trainset.append(data_dict["train%d"%i])
        traintargets.append(np.full(len(data_dict["train%d"%i]),i))
        testset.append(data_dict["test%d"%i])
        testtargets.append(np.full(len(data_dict["test%d"%i]),i))
    
    #Concatenate into to complete datasets
    trainset = np.concatenate(trainset)
    traintargets = np.concatenate(traintargets)
    testset = np.concatenate(testset)
    testtargets = np.concatenate(testtargets)
    return trainset, traintargets, testset, testtargets
# def likelihood(data, mean, cov):
#    likelihood_value = multivariate_normal.pdf(data, mean, cov) 
#    return likelihood_value
# def func(data, mean, cov):
#     likelihoods = []
#     test = []
#     for j in data:
#         test.append(likelihood(j, mean, cov))
#     likelihoods.append(test)
#     return likelihoods
# file = "Lecture_6/mnist_all.mat"

def SVM_test(train_set, train_targets, test_set, test_targets):
    decomp = svm.SVC()
    trnSVM = decomp.fit(train_set, train_targets)
    test_SVM = decomp.predict(test_set)

    counter = 0
    for i in range(len(test_SVM)):
        if  test_SVM[i] == int(test_targets[i]):
            counter += 1
    acc_svm = counter/len(test_targets)
    print(f"The accuracy for the MNIST dataset with SVM is: {acc_svm*100:.2f}%")
    cm = confusion_matrix(test_SVM, test_targets)
    cmplot = ConfusionMatrixDisplay(cm)
    cmplot.plot()
    plt.show()
    return

def main():
    file = "C:/Users/jakob/Documents/AAU/7_semester/machineLearning/pythonCode/7semML/Lecture_6/mnist_all.mat"
    data = loadmat(file)
    #Complete training and test sets
    t0 = time.time()
    train_set, train_targets, test_set, test_targets = create_complete_datasets(data)
    SVM_test(train_set, train_targets, test_set, test_targets)
    t1 = time.time()
    time_svm = t1-t0
    print(f"Time to run code is {time_svm}") 

if __name__ == "__main__":
    main()


# print(counter)

# Use sklearn
# SVM_test()

# ## Test model on test set


# What is the accuracy on the test set?


# ## Plot Confusion matrix


# Does the confusion matrix show us any insights about the model perfromance?


# ## Comparing with PCA/LDA
# 


# How does SVM compare to PCA and LDA (compare confusion matrices)


