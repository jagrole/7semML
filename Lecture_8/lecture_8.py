# ###################################
# Group ID : 985
# Members : Jakob G. Olesen, Mads Lindeborg Larsen, Sif Bjerre Lindby
# Date : October 18 2023
# Lecture: 8 Multilayer Perceptrons
# Dependencies: numpy, scipy.stats, scipy.io, matplotlib.pyplot, sklearn
# Python version: 3.11.3
# Functionality: This script performs classification using MLP on data decomposed with LDA and PCA respectively on the entire mnist data set
# ###################################
import numpy as np
from scipy.io import loadmat 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.neural_network import MLPClassifier as MLPC 

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

# #Create LDA data
# decomp_lda = LDA(n_components=9)
# trnLDA = decomp_lda.fit_transform(train_set, train_targets)
# testLDA = decomp_lda.transform(test_set)

#Create PCA data
def create_pca_decomp(train_set, test_set, pca_dim):
    decomp_pca = PCA(n_components=pca_dim)
    trnPCA = decomp_pca.fit_transform(train_set)
    testPCA = decomp_pca.transform(test_set)
    return trnPCA, testPCA


#MLP LDA 
# MLP_lda_test = MLPC(max_iter = 500).fit(trnLDA, train_targets) 
# test_LDA_MLP = MLP_lda_test.predict(testLDA)
# counter = 0
# for i in range(len(test_LDA_MLP)):
#     if  test_LDA_MLP[i] == int(test_targets[i]):
#         counter += 1
# acc_LDA_MLP = counter/len(test_targets)
# cm_LDA_MLP = confusion_matrix(test_LDA_MLP, test_targets)
# cm_mlp = confusion_matrix(test_LDA_MLP, test_targets)
# cmplot = ConfusionMatrixDisplay(cm_LDA_MLP)
# cmplot.plot()
# plt.show()

#MLP PCA
def main():  
    file = "C:/Users/jakob/Documents/AAU/7_semester/machineLearning/pythonCode/7semML/Lecture_6/mnist_all.mat"
    data = loadmat(file)
    train_set, train_targets, test_set, test_targets = create_complete_datasets(data)
    pca_dim = [10,20,30]
    for j in pca_dim:
        trnPCA, testPCA = create_pca_decomp(train_set, test_set, j)
        MLP_PCA_test = MLPC(hidden_layer_sizes=[100,100], max_iter = 500).fit(trnPCA, train_targets) 
        test_PCA_MLP = MLP_PCA_test.predict(testPCA)
        counter = 0
        for i in range(len(test_PCA_MLP)):
            if  test_PCA_MLP[i] == int(test_targets[i]):
                counter += 1
        acc_PCA_MLP = counter/len(test_targets)
        cm_PCA_MLP = confusion_matrix(test_PCA_MLP, test_targets)
        cm_mlp = confusion_matrix(test_PCA_MLP, test_targets)
        cmplot = ConfusionMatrixDisplay(cm_PCA_MLP)
        cmplot.plot()
        plt.show()
        print(f"The accuracy for the MNIST dataset with MLP with PCA of dimensionality {j} decomposed data is: {acc_PCA_MLP*100:.2f}%")

if __name__ == "__main__":
    main()
