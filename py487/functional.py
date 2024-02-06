import numpy as np

def kl_div(input,target):
    """
    input: numpy.array of arbitrary shape in log-probabilities
    target: numpy.array of the same shape as input (not in log-probabilities)
    """
    kl = 0
    for i in range(len(target)):
        for j in range(len(target.columns)):
            ep = 1e-10
            kl += target.values[i][j] * np.log((target.values[i][j] + ep) / (input.values[i][j] + ep))
    return kl/len(target)

def cross_entropy(input,target):
    """
    input: numpy.array of arbitrary shape in log-probabilities
    target: numpy.array of the same shape as input (not in log-probabilities)
    """
    ce = 0
    for i in range(len(target)):
        for j in range(len(target.columns)):
            ep = 1e-10
            ce += target.values[i][j] * np.log(input.values[i][j] + ep)
    return -ce/len(target)

def minimize_gradient_descent(gradient_funcs,alpha,theta0,tol=1e-10):
    """
    You can stop optimizing when the change in theta is < tol
    """
    thetas = [theta0] * len(gradient_funcs)
    while True:
        gradients = [g(theta) for g, theta in zip(gradient_funcs, thetas)]
        newThetas = [theta - alpha * grad for theta, grad in zip(thetas, gradients)]
        if np.all(np.abs(np.array(thetas) - np.array(newThetas)) < tol):
            thetas = newThetas
            break
        thetas = newThetas
    
    return thetas

def minimize_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=100,debug=True):
    """
    You can stop optimizing when the change in theta is < tol
    """
    thetas = [theta0]
    for i in range(max_iter):
        newThetas = []
        for theta in thetas:
            j_grad = (J_func(thetas[0] + h) - J_func(thetas[0])) / h
            newTheta = theta - alpha * j_grad
            newThetas.append(newTheta)
        if np.all(np.abs(np.array(thetas) - np.array(newThetas)) < tol):
            thetas = newThetas
            break
        thetas = newThetas
    return thetas