# ###################################
# Group ID : 985
# Members : Jakob G. Olesen, Mads Lindeborg Larsen, Sif Bjerre Lindby
# Date : September 20 2023
# Lecture: 4 Dimensionality reduction
# Dependencies: numpy, scipy.stats, matplotlib.pyplot, sklearn.decomposition, sklearn.discriminant_analysis
# Python version: 3.11.3
# Functionality: This script classifies handwritten digits as either 5, 6, or 8 using 2 
# different dimensionality reduction methods: PCA and LDA.
# ###################################


import numpy as np
from scipy.stats import multivariate_normal 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#Functions for later use

def likelihood(data, mean, cov):
   likelihood_value = multivariate_normal.pdf(data, mean, cov) 
   return likelihood_value

def classify_data(data1, data2, data3):
   max_index_xy_list = []
   for entry in zip(data1, data2, data3):
      max_index_xy_list.append(np.argmax(entry))
   return max_index_xy_list

def correct_counter(data, cla):
   correct = 0
   for entry in data:
         if entry == cla:
            correct+= 1
   return correct

def main():
   # Loading the training and test data, /255 for normalization
   train5 = np.loadtxt("mnist_all/train5.txt")/255 
   train6 = np.loadtxt("mnist_all/train6.txt")/255
   train8 = np.loadtxt("mnist_all/train8.txt")/255
   test5 = np.loadtxt("mnist_all/test5.txt")/255
   test6 = np.loadtxt("mnist_all/test6.txt")/255
   test8 = np.loadtxt("mnist_all/test8.txt")/255 

   # Define targets 
   train6_target = 6*np.ones(len(train6))
   train8_target = 8*np.ones(len(train8))  
   train5_target = 5*np.ones(len(train5))
   test5_target = 5*np.ones(len(test5))
   test6_target = 6*np.ones(len(test6))
   test8_target = 8*np.ones(len(test8))

   # # Combine data
   train_data = np.concatenate([train5, train6, train8])
   train_targets = np.concatenate([train5_target, train6_target, train8_target])

   #PCA model fit
   decomp = PCA(n_components=2)
   decomp.fit(train_data)

   x5PCA = decomp.transform(train5)
   x6PCA = decomp.transform(train6)
   x8PCA = decomp.transform(train8)

   x5PCA_test = decomp.transform(test5)
   x6PCA_test = decomp.transform(test6)
   x8PCA_test = decomp.transform(test8)


   #PCA mean, variance and Covariance matrices
   trn_mean_x5_PCA = np.mean(x5PCA, axis= 0)
   trn_var_x5_PCA = np.var(x5PCA, axis= 0)
   trn_cov_x5_PCA = np.cov(x5PCA.T)

   trn_mean_x6_PCA = np.mean(x6PCA, axis= 0)
   trn_var_x6_PCA = np.var(x6PCA, axis= 0)
   trn_cov_x6_PCA = np.cov(x6PCA.T)

   trn_mean_x8_PCA = np.mean(x8PCA, axis= 0)
   trn_var_x8_PCA = np.var(x8PCA, axis= 0)
   trn_cov_x8_PCA = np.cov(x8PCA.T)

   #LDA decomposition and model fit
   decompLDA = LDA(n_components=2)
   decompLDA.fit(train_data, train_targets)

   x5LDA = decompLDA.transform(train5)
   x6LDA = decompLDA.transform(train6)
   x8LDA = decompLDA.transform(train8)

   x5LDA_test = decompLDA.transform(test5)
   x6LDA_test = decompLDA.transform(test6)
   x8LDA_test = decompLDA.transform(test8)

   #LDA mean, variance and Covariance matrices
   trn_mean_x5_LDA = np.mean(x5LDA, axis= 0)
   trn_var_x5_LDA = np.var(x5LDA, axis= 0)
   trn_cov_x5_LDA = np.cov(x5LDA.T)

   trn_mean_x6_LDA = np.mean(x6LDA, axis= 0)
   trn_var_x6_LDA = np.var(x6LDA, axis= 0)
   trn_cov_x6_LDA = np.cov(x6LDA.T)

   trn_mean_x8_LDA = np.mean(x8LDA, axis= 0)
   trn_var_x8_LDA = np.var(x8LDA, axis= 0)
   trn_cov_x8_LDA = np.cov(x8LDA.T)

   prior_x5 = len(train5)/(len(train5)+len(train8)+len(train6))
   prior_x6 = len(train6)/(len(train5)+len(train8)+len(train6))
   prior_x8 = len(train8)/(len(train5)+len(train8)+len(train6))
   # X5 PCA
   # Compute likelihoods of x5
   likelihood_x5_in_x5_PCA = [] 
   for entry in x5PCA_test:
         likelihood_x5_in_x5_PCA.append(likelihood(entry, trn_mean_x5_PCA, trn_cov_x5_PCA))  
   likelihood_x5_in_x6_PCA= [] 
   for entry in x5PCA_test:
         likelihood_x5_in_x6_PCA.append(likelihood(entry, trn_mean_x6_PCA, trn_cov_x6_PCA))  
   likelihood_x5_in_x8_PCA = []  
   for entry in x5PCA_test:
        likelihood_x5_in_x8_PCA.append(likelihood(entry, trn_mean_x8_PCA, trn_cov_x8_PCA))  
   
   # Compute Posteriors 
   posterior_x5_in_x5_PCA = [x * prior_x5 for x in likelihood_x5_in_x5_PCA]
   posterior_x5_in_x6_PCA = [y * prior_x6 for y in likelihood_x5_in_x6_PCA]
   posterior_x5_in_x8_PCA = [z * prior_x8 for z in likelihood_x5_in_x8_PCA]
   
   max_index_list_PCA_x5 = classify_data(posterior_x5_in_x5_PCA, posterior_x5_in_x6_PCA, posterior_x5_in_x8_PCA)
   correct_x5_PCA = correct_counter(max_index_list_PCA_x5, 0)
   accuracy_x5_PCA = correct_x5_PCA/len(test5_target)
   print(f"The accuracy for x5 with PCA is: {accuracy_x5_PCA*100:.2f}%")

# X6 PCA
# Compute likelihoods of x6
   likelihood_x6_in_x5_PCA = [] 
   for entry in x6PCA_test:
         likelihood_x6_in_x5_PCA.append(likelihood(entry, trn_mean_x5_PCA, trn_cov_x5_PCA))  
   likelihood_x6_in_x6_PCA= [] 
   for entry in x6PCA_test:
         likelihood_x6_in_x6_PCA.append(likelihood(entry, trn_mean_x6_PCA, trn_cov_x6_PCA))  
   likelihood_x6_in_x8_PCA = []  
   for entry in x6PCA_test:
        likelihood_x6_in_x8_PCA.append(likelihood(entry, trn_mean_x8_PCA, trn_cov_x8_PCA))  
   
   # Compute Posteriors 
   posterior_x6_in_x5_PCA = [x * prior_x5 for x in likelihood_x6_in_x5_PCA]
   posterior_x6_in_x6_PCA = [y * prior_x6 for y in likelihood_x6_in_x6_PCA]
   posterior_x6_in_x8_PCA = [z * prior_x8 for z in likelihood_x6_in_x8_PCA]

   max_index_list_PCA_x6 = classify_data(posterior_x6_in_x5_PCA, posterior_x6_in_x6_PCA, posterior_x6_in_x8_PCA)

   # Compute the accuracy of our classifications 
   correct_x6_PCA = correct_counter(max_index_list_PCA_x6, 1)
   accuracy_x6_PCA = correct_x6_PCA/len(test6_target)
   print(f"The accuracy for x6 with PCA is: {accuracy_x6_PCA*100:.2f}%")
   # X8 PCA
   # Compute likelihoods of x8
   likelihood_x8_in_x5_PCA = [] 
   for entry in x8PCA_test:
         likelihood_x8_in_x5_PCA.append(likelihood(entry, trn_mean_x5_PCA, trn_cov_x5_PCA))  
   likelihood_x8_in_x6_PCA= [] 
   for entry in x8PCA_test:
         likelihood_x8_in_x6_PCA.append(likelihood(entry, trn_mean_x6_PCA, trn_cov_x6_PCA))  
   likelihood_x8_in_x8_PCA = []  
   for entry in x8PCA_test:
        likelihood_x8_in_x8_PCA.append(likelihood(entry, trn_mean_x8_PCA, trn_cov_x8_PCA))  

   # Compute Posteriors     
   posterior_x8_in_x5_PCA = [x * prior_x5 for x in likelihood_x8_in_x5_PCA]
   posterior_x8_in_x6_PCA = [y * prior_x6 for y in likelihood_x8_in_x6_PCA]
   posterior_x8_in_x8_PCA = [z * prior_x8 for z in likelihood_x8_in_x8_PCA]

   max_index_list_PCA_x8 = classify_data(posterior_x8_in_x5_PCA, posterior_x8_in_x6_PCA, posterior_x8_in_x8_PCA)

   # Compute the accuracy of our classifications
   correct_x8_PCA = correct_counter(max_index_list_PCA_x8, 2)
   accuracy_x8_PCA = correct_x8_PCA/len(test8_target)
   print(f"The accuracy for x8 with PCA is: {accuracy_x8_PCA*100:.2f}%")


#  X5 LDA
   # Compute likelihoods of x5
   likelihood_x5_in_x5_LDA = [] 
   for entry in x5LDA_test:
         likelihood_x5_in_x5_LDA.append(likelihood(entry, trn_mean_x5_LDA, trn_cov_x5_LDA))  
   likelihood_x5_in_x6_LDA= [] 
   for entry in x5LDA_test:
         likelihood_x5_in_x6_LDA.append(likelihood(entry, trn_mean_x6_LDA, trn_cov_x6_LDA))  
   likelihood_x5_in_x8_LDA = []  
   for entry in x5LDA_test:
        likelihood_x5_in_x8_LDA.append(likelihood(entry, trn_mean_x8_LDA, trn_cov_x8_LDA))   
  
   # Compute Posteriors

   posterior_x5_in_x5_LDA = [x * prior_x5 for x in likelihood_x5_in_x5_LDA]
   posterior_x5_in_x6_LDA = [y * prior_x6 for y in likelihood_x5_in_x6_LDA]
   posterior_x5_in_x8_LDA = [z * prior_x8 for z in likelihood_x5_in_x8_LDA]


   # Classify test data

   max_index_list_LDA_x5 = classify_data(posterior_x5_in_x5_LDA, posterior_x5_in_x6_LDA, posterior_x5_in_x8_LDA)

   # Compute the accuracy of our classifications 
   correct_x5_LDA = correct_counter(max_index_list_LDA_x5, 0)
   accuracy_x5_LDA = correct_x5_LDA/len(test5_target)
   print(f"The accuracy for x5 with LDA is: {accuracy_x5_LDA*100:.2f}%")

   #  X6 LDA
   # Compute likelihood of x and y being in the correct sets
   likelihood_x6_in_x5_LDA = [] 
   for entry in x6LDA_test:
         likelihood_x6_in_x5_LDA.append(likelihood(entry, trn_mean_x5_LDA, trn_cov_x5_LDA))  
   likelihood_x6_in_x6_LDA= [] 
   for entry in x6LDA_test:
         likelihood_x6_in_x6_LDA.append(likelihood(entry, trn_mean_x6_LDA, trn_cov_x6_LDA))  
   likelihood_x6_in_x8_LDA = []  
   for entry in x6LDA_test:
        likelihood_x6_in_x8_LDA.append(likelihood(entry, trn_mean_x8_LDA, trn_cov_x8_LDA))   
  
   # We compute the posterior probability by taking the priors into account

   posterior_x6_in_x5_LDA = [x * prior_x5 for x in likelihood_x6_in_x5_LDA]
   posterior_x6_in_x6_LDA = [y * prior_x6 for y in likelihood_x6_in_x6_LDA]
   posterior_x6_in_x8_LDA = [z * prior_x8 for z in likelihood_x6_in_x8_LDA]


   # Classify test data

   max_index_list_LDA_x6 = classify_data(posterior_x6_in_x5_LDA, posterior_x6_in_x6_LDA, posterior_x6_in_x8_LDA)
   # Compute the accuracy of our classifications 
   correct_x6_LDA = correct_counter(max_index_list_LDA_x6, 1)
   accuracy_x6_LDA = correct_x6_LDA/len(test6_target)
   print(f"The accuracy for x6 with LDA is: {accuracy_x6_LDA*100:.2f}%")

      #  X8 LDA
   # Compute likelihood of x and y being in the correct sets
   likelihood_x8_in_x5_LDA = [] 
   for entry in x8LDA_test:
         likelihood_x8_in_x5_LDA.append(likelihood(entry, trn_mean_x5_LDA, trn_cov_x5_LDA))  
   likelihood_x8_in_x6_LDA= [] 
   for entry in x8LDA_test:
         likelihood_x8_in_x6_LDA.append(likelihood(entry, trn_mean_x6_LDA, trn_cov_x6_LDA))  
   likelihood_x8_in_x8_LDA = []  
   for entry in x8LDA_test:
        likelihood_x8_in_x8_LDA.append(likelihood(entry, trn_mean_x8_LDA, trn_cov_x8_LDA))   
  
   # We compute the posterior probability by taking the priors into account

   posterior_x8_in_x5_LDA = [x * prior_x5 for x in likelihood_x8_in_x5_LDA]
   posterior_x8_in_x6_LDA = [y * prior_x6 for y in likelihood_x8_in_x6_LDA]
   posterior_x8_in_x8_LDA = [z * prior_x8 for z in likelihood_x8_in_x8_LDA]


  # Classify test data

   max_index_list_LDA_x8 = classify_data(posterior_x8_in_x5_LDA, posterior_x8_in_x6_LDA, posterior_x8_in_x8_LDA)

   # Compute the accuracy of our classifications
   correct_x8_LDA = correct_counter(max_index_list_LDA_x8, 2)
   accuracy_x8_LDA = correct_x8_LDA/len(test8_target)
   print(f"The accuracy for x8 with LDA is: {accuracy_x8_LDA*100:.2f}%")
if __name__ == "__main__":
    main()

