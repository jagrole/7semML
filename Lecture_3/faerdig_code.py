 ###################################
# Group ID : <gr oup_id >
# Members : Sif Bjerre Lindby, Mads Lindeborg Larsen, Jakob Olavi Grangaard Olesen
# Date : 15/9 - 2023
# Lecture: 3 Parametric and nonparametric methods 
# Dependencies: numpy, scipy
# Python version: 3.11.2
# Functionality: This script computes and classifies some numbers according to a given training set. It is split into 3 main functions according to assignment a, b and c.
# ##################################
import numpy as np
from scipy.stats import multivariate_normal


# Load Training Data
trn_x = np.loadtxt("trn_x.txt")
trn_x_label = np.loadtxt("trn_x_class.txt")

trn_y = np.loadtxt("trn_y.txt")
trn_y_label = np.loadtxt("trn_y_class.txt")

# Load Testing Data
tst_x = np.loadtxt("tst_x.txt")
tst_x_label = np.loadtxt("tst_x_class.txt")

tst_y = np.loadtxt("tst_y.txt")
tst_y_class = np.loadtxt("tst_y_class.txt")

tst_y_126 = np.loadtxt("tst_y_126.txt")
tst_y_126_class = np.loadtxt("tst_y_126_class.txt")

tst_xy = np.loadtxt("tst_xy.txt")
tst_xy_class = np.loadtxt("tst_xy_class.txt")

tst_xy_126 = np.loadtxt("tst_xy_126.txt")
tst_xy_126_class = np.loadtxt("tst_xy_126_class.txt")

# Compute the parameters of x and y using built-in numpy functions.

# Trained x parameters

trn_x_mean = np.mean(trn_x, axis= 0)
train_var = np.var(trn_x, axis= 0)
trn_x_cov = np.cov(trn_x.T)

# Trained y parameters

trn_y_mean = np.mean(trn_y, axis = 0)
trn_y_cov = np.cov(trn_y.T)

# Likelihood function taken from the multivariate module and some functions for later use. 


def likelihood(data, mean, cov):
   likelihood_value = multivariate_normal.pdf(data, mean, cov)
   return likelihood_value

def likelihood_of_xy_in_x_y(data):
      likelihood_xy_in_x = []
      for entry in data:
         likelihood_xy_in_x.append(likelihood(entry, trn_x_mean, trn_x_cov))
      likelihood_xy_in_y = []
      for entry in data:
         likelihood_xy_in_y.append(likelihood(entry, trn_y_mean, trn_y_cov))
      return likelihood_xy_in_x, likelihood_xy_in_y

def classify_data(data1, data2):
   max_index_xy_list = []
   for entry in zip(data1, data2):
      max_index_xy_list.append(np.argmax(entry))
   return max_index_xy_list

def compute_correct_classification(max_index_list, test_data, target_data, class_list):
   correct_classification_counter = 0
   for entry, i in zip(test_data, enumerate(test_data)):
      for entry2, _ in zip(target_data, enumerate(target_data)):
         if np.array_equiv(entry,entry2) and max_index_list[i[0]]+1 == class_list[i[0]]:
            correct_classification_counter += 1
   return correct_classification_counter


def assignment_a():
   # Compute the Priors as defined by the training data.
   prior_x = len(trn_x)/(len(trn_x)+len(trn_y))
   prior_y = 1-prior_x

   # Compute likelihood of x and y being in the correct sets

   likelihood_xy_in_x, likelihood_xy_in_y = likelihood_of_xy_in_x_y(tst_xy)

   # We compute the posterior probability by taking the priors into account

   posterior_xy_in_x = [x * prior_x for x in likelihood_xy_in_x]
   posterior_xy_in_y = [y * prior_y for y in likelihood_xy_in_y]
   

   # Classify test data as belonging to the class with the highest posterior probability

   max_index_xy_list = classify_data(posterior_xy_in_x, posterior_xy_in_y)

   # Compute the accuracy of our classifications by taking the sum of correct predictions and divide by the total number of predictions
   x_from_xy = []
   for entry in range(len(tst_xy)):
      if max_index_xy_list[entry] == 0:
         x_from_xy.append(tst_xy[entry])
   y_from_xy = []
   for entry in range(len(tst_xy)):
      if max_index_xy_list[entry] == 1:
         y_from_xy.append(tst_xy[entry])
   
   correct_x = compute_correct_classification(max_index_xy_list, x_from_xy, tst_xy, tst_xy_class)
   correct_y = compute_correct_classification(max_index_xy_list, y_from_xy, tst_xy, tst_xy_class)
   accuracy = (correct_x + correct_y)/len(tst_xy_class)

   print(f"The accuracy for assignment a) is: {accuracy*100:.2f}%")
   return

def assignment_b():

# First we define our prior probabilities and likelihoods
   prior_x_uniform = 0.5
   prior_y_uniform = 0.5
   likelihood_xy_in_x, likelihood_xy_in_y = likelihood_of_xy_in_x_y(tst_xy_126)

# We can now compute posteriors knowing that the posterior probability is simply the prior, p(C), multiplied by the likelihood p(x, C).
   posterior_xy_in_x = [x * prior_x_uniform for x in likelihood_xy_in_x]
   posterior_xy_in_y = [y * prior_y_uniform for y in likelihood_xy_in_y]
   max_index_xy_list = classify_data(posterior_xy_in_x, posterior_xy_in_y)
   

# Now that we have posteriors for both x and y we can classify the test data and compute the accuracy
   x_from_xy = []
   for entry in range(len(tst_xy_126)):
      if max_index_xy_list[entry] == 0:
         x_from_xy.append(tst_xy_126[entry])
   y_from_xy = []
   for entry in range(len(tst_xy_126)):
      if max_index_xy_list[entry] == 1:
         y_from_xy.append(tst_xy_126[entry])

   correct_x = compute_correct_classification(max_index_xy_list, x_from_xy, tst_xy_126, tst_xy_126_class)
   correct_y = compute_correct_classification(max_index_xy_list, y_from_xy, tst_xy_126, tst_xy_126_class)
   
   accuracy = (correct_x + correct_y)/len(tst_xy_126_class)

   print(f"The accuracy for assignment b) is: {accuracy*100:.2f}%")
   return

# (c) classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).
def assignment_c():

# First we define our prior probabilities and likelihoods
   prior_x_c = 0.9
   prior_y_c = 0.1
   likelihood_xy_in_x, likelihood_xy_in_y = likelihood_of_xy_in_x_y(tst_xy_126)

# We can now compute posteriors knowing that the posterior probability is simply the prior, p(C), multiplied by the likelihood p(x, C).
   posterior_xy_in_x = [x * prior_x_c for x in likelihood_xy_in_x]
   posterior_xy_in_y = [y * prior_y_c for y in likelihood_xy_in_y]
   max_index_xy_list = classify_data(posterior_xy_in_x, posterior_xy_in_y)
   # print(max_index_xy_list)

# Now that we have posteriors for both x and y we can classify the test data and compute the accuracy
   x_from_xy = []
   for entry in range(len(tst_xy_126)):
      if max_index_xy_list[entry] == 0:
         x_from_xy.append(tst_xy_126[entry])
   y_from_xy = []
   for entry in range(len(tst_xy_126)):
      if max_index_xy_list[entry] == 1:
         y_from_xy.append(tst_xy_126[entry])

   correct_x = compute_correct_classification(max_index_xy_list, x_from_xy, tst_xy_126, tst_xy_126_class)
   correct_y = compute_correct_classification(max_index_xy_list, y_from_xy, tst_xy_126, tst_xy_126_class)

   accuracy = (correct_x + correct_y)/len(tst_xy_126_class)
   print(f"The accuracy for assignment c) is: {accuracy*100:.2f}%")
   return


assignment_a()
assignment_b()
assignment_c()