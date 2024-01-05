import numpy as np

def kl_div(input,target):
    """
    input: numpy.array of arbitrary shape in log-probabilities
    target: numpy.array of the same shape as input (not in log-probabilities)
    """
    return 0

def cross_entropy(input,target):
    """
    input: numpy.array of arbitrary shape in log-probabilities
    target: numpy.array of the same shape as input (not in log-probabilities)
    """
    return 0

def minimize_gradient_descent(gradient_funcs,alpha,theta0,tol=1e-10):
    """
    You can stop optimizing when the change in theta is < tol
    """
    thetas = [theta0]
    return thetas

def minimize_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=100,debug=True):
    """
    You can stop optimizing when the change in theta is < tol
    """
    thetas = [theta0]
    return thetas