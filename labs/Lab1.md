# Lab 1 - Loss functions
We are not at the point in this class where we have covered network architectures, gradient descent, and many more topics. We are familiar with the concept of a loss function. We know there are different options when choosing a loss function. This lab is to designed to expose you to PyTorch through an exploration of loss functions. You are not expected to code any new PyTorch NN code. You are expected to apply your Python programming skills to complete this lab (i.e., no need to change the model itself or the training code significantly). 

## Overall Instructions
With a partner, answer the following questions and respond to the prompts. Upload your work as a PDF file to **both** of your Canvas accounts. You can use the materials in class and the internet to answer the questions and complete the prompts. Only typset answers are accepted (i.e., nothing hand written).

## Questions

1. Compare and contrast Hinge Loss, Cross Entropy Loss, and Mean Squared Error.
3. What is regularization and why is it needed?

## Exercise

### Motivation
Loss functions are very important in deep learning. In this lab, we are going to explore how to answer the following questions: 

* How does a data scientist decide one loss function is better for their data? 
* How do they evaluate two potential loss functions? 
* What kinds of datasets are most suitable for a particular loss function?
* How do we automate these tasks?
* How do we present our findings in a notebook?

### Instructions
You should work with your partner, but you should have your own copy of work that was generated with your own hands (i.e., this is not pair programming).

1. Make a copy of this notebook: https://colab.research.google.com/drive/1p-rNg1y7IhKFUaWPRIfIbuDlvo_3FyD8?usp=sharing
2. Write a description of each code cell in Markdown cells above the code cells.
3. Modify the notebook so that it compares two loss functions using the dataset provided in the notebook (cross-entropy and mean squared error). Your notebook should provide a reproducible experiment. 
4. Write up a synthesis of the results and what they mean for this data. Did one loss perform better? By how much? Is this conclusive? etc.
5. Extend the notebook on a new dataset of your choosing such that your conclusions about which loss function performs better is reversed.
6. Upload your findings to Canvas.