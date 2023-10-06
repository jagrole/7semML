import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
def likelihood(data, mean, cov):
   likelihood_value = multivariate_normal.pdf(data, mean, cov) 
   return likelihood_value
def func(data, mean, cov):
    likelihoods = []
    test = []
    for j in data:
        test.append(likelihood(j, mean, cov))
    likelihoods.append(test)
    return likelihoods
file = "mnist_all.mat"
data = loadmat(file)

#Complete training and test sets
train_set1, train_targets, test_set1, test_targets = create_complete_datasets(data)
train_set = train_set1/255
test_set = test_set1/255

def PCA_test():
    decomp = PCA(n_components=9)
    trnPCA = decomp.fit_transform(train_set)
    test_PCA = decomp.transform(test_set)
    trainPCA = {str(i): trnPCA[train_targets == (i)] for i in range(10)}

    mean_cov_priors_dict = {} 
    for i in range(10):
        tempstr = str(i)
        currentdata = trainPCA[tempstr]
        currentmean = np.mean(currentdata, axis = 0)
        currentcov = np.cov(currentdata.T)
        prior = len(currentdata)/60000
        temp = [currentmean, currentcov, prior]
        mean_cov_priors_dict[tempstr] = temp

    testasd = []
    for i in range(10):
        tempstr = str(i)
        tempval = mean_cov_priors_dict[tempstr]
        test123 = func(test_PCA, tempval[0], tempval[1])
        test123real = [x * tempval[2] for x in test123[0]]
        testasd.append(test123real)


    asd2 = np.array(testasd)
    clfsy = np.argmax(asd2, axis = 0)

    counter = 0
    for i in range(len(clfsy)):
        if  clfsy[i] == int(test_targets[i]):
            counter += 1

    acc = counter/len(test_targets)
    print(f"The accuracy for the MNIST dataset with PCA is: {acc*100:.2f}%")
    cm = confusion_matrix(clfsy, test_targets)
    cmplot = ConfusionMatrixDisplay(cm)
    cmplot.plot()
    plt.show()
    return

###############################################################################################
def LDA_test():
    #LDAdecomp
    decomp = LDA(n_components=9)
    trnLDA = decomp.fit_transform(train_set, train_targets)
    test_LDA = decomp.transform(test_set)
    trainLDA = {str(i): trnLDA[train_targets == (i)] for i in range(10)}
    
    mean_cov_priors_dict = {} 
    for i in range(10):
        tempstr = str(i)
        currentdata = trainLDA[tempstr]
        currentmean = np.mean(currentdata, axis = 0)
        currentcov = np.cov(currentdata.T)
        prior = len(currentdata)/60000
        temp = [currentmean, currentcov, prior]
        mean_cov_priors_dict[tempstr] = temp
        
    testasd = []
    for i in range(10):
        tempstr = str(i)
        tempval = mean_cov_priors_dict[tempstr]
        test123 = func(test_LDA, tempval[0], tempval[1])
        test123real = [x * tempval[2] for x in test123[0]]
        testasd.append(test123real)

    asd2 = np.array(testasd)
    clfsy = np.argmax(asd2, axis = 0)

    counter = 0
    for i in range(len(clfsy)):
        if  clfsy[i] == int(test_targets[i]):
            counter += 1

    acc_lda = counter/len(test_targets)
    print(f"The accuracy for the MNIST dataset with LDA is: {acc_lda*100:.2f}%")
    cm = confusion_matrix(clfsy, test_targets)
    cmplot = ConfusionMatrixDisplay(cm)
    cmplot.plot()
    plt.show()
    return 

PCA_test()
LDA_test()
