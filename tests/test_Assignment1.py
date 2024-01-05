import sys
import os
import torch

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

sys.path.insert(0,f'{DIR}/../')

import py487

import joblib 
answers = joblib.load(str(DIR)+"/answers_Assignment1.joblib")

import pandas as pd
import numpy as np

from sklearn import datasets
np.random.seed(4)
X, t_fruit = datasets.make_blobs(n_samples=100, centers=3, n_features=2, center_box=(0, 10))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, t_fruit)
y_fruit_probs_pred = pd.DataFrame(clf.predict_proba(X),columns=clf.classes_)
y_fruit_probs = pd.get_dummies(t_fruit).astype(float)

def test_1():
    solution = float(torch.nn.functional.kl_div(torch.tensor(np.log(y_fruit_probs_pred.values+1e-10)), torch.tensor(y_fruit_probs.values),reduction='batchmean').numpy())
    answer = py487.functional.kl_div(np.log(y_fruit_probs_pred.values+1e-10),y_fruit_probs.values)
    assert np.abs(solution-answer) <= 0.0001
    
def test_2():    
    solution = float(torch.nn.functional.cross_entropy(torch.tensor(np.log(y_fruit_probs_pred.values+1e-10)), torch.tensor(y_fruit_probs.values)).numpy())
    answer = py487.functional.cross_entropy(np.log(y_fruit_probs_pred.values+1e-10),y_fruit_probs.values)
    assert np.abs(solution-answer) <= 0.0001
    
def test_3():    
    solution_thetas = answers['minimize_gradient_descent']
    gradient_func1 = lambda theta: 2*theta
    gradient_func2 = lambda theta: -2*(1-theta)
    answer_thetas = py487.functional.minimize_gradient_descent([gradient_func1,gradient_func2],0.1,5)
    assert np.all(np.abs(np.array(solution_thetas)-np.array(answer_thetas)) <= 0.0001)
    
def test_4():    
    solution_thetas = answers['minimize_gradient_descent_analytically']
    J1_func = lambda theta: theta**2
    J2_func = lambda theta: (1-theta)**2
    R_func = lambda theta: 1/2*(J1_func(theta)+J2_func(theta))
    answer_thetas = py487.functional.minimize_gradient_descent_analytically(R_func,0.1,5,0.01)
    assert np.all(np.abs(np.array(solution_thetas)-np.array(answer_thetas)) <= 0.0001)