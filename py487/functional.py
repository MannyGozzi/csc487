import numpy as np

def kl_div(input,target):
    """
    input: numpy.array of arbitrary shape in log-probabilities
    target: numpy.array of the same shape as input (not in log-probabilities)
    """
    ## BEGIN SOLUTION
    return -np.sum(target*(input - np.log(target+1e-10)))/len(target)
    ## END SOLUTION
    return 0

def cross_entropy(input,target):
    """
    input: numpy.array of arbitrary shape in log-probabilities
    target: numpy.array of the same shape as input (not in log-probabilities)
    """
    ## BEGIN SOLUTION
    return -np.sum(target*input)/len(target)
    ## END SOLUTION
    return 0

def minimize_gradient_descent(gradient_funcs,alpha,theta0,tol=1e-10):
    """
    You can stop optimizing when the change in theta is < tol
    """
    thetas = [theta0]
    ## BEGIN SOLUTION
    theta = theta0
    while True:
        theta_before = theta
        theta = theta - alpha*1/2*np.sum([f(theta) for f in gradient_funcs])
        if np.abs(theta - theta_before) < tol:
            break
        thetas.append(theta)
    ## END SOLUTION    
    return thetas

def minimize_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=100,debug=True):
    """
    You can stop optimizing when the change in theta is < tol
    """
    thetas = [theta0]
    ## BEGIN SOLUTION
    theta = theta0
    c = 0
    grad = None
    if debug:
        print("iteration,gradient,theta,previous theta")
    while c < max_iter:
        theta_before = theta
        grad_before = grad
        grad = (J_func(theta+h)-J_func(theta))/h
        if grad is not None and grad_before is not None:
            if np.sign(grad) != np.sign(grad_before):
                print('Gradient swapped.')
                break
        theta = theta - alpha*grad
        if np.abs(theta - theta_before) < tol:
            break
        thetas.append(theta)
        if debug and c % 10 == 0:
            print(c,grad,theta,theta_before,sep=",")
        c+=1
    ## END SOLUTION    
    return thetas