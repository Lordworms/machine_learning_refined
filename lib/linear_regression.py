import matplotlib.pyplot as plt
import numpy as np
def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def g(w0,w1,xp):
    return np.exp(w0+np.log(xp)*w1)


def softMaxCost(w,y,model,x_b,lam=10**-3):
    p=y.size
    a=1+np.exp(-y*model(x_b,w))
    cost=(1/p)*np.sum(np.log(a))
    cost+=lam*np.sum(w[1:]**2)
    return cost


def gradient_descent(x, y, w, cost, gradient, alpha=0.1, max_its=100):
    weight_history = [w]          
    cost_history = [cost(w)]
    for _ in range(max_its):
        grad_eval = gradient(w)
        w = w - alpha*grad_eval
        weight_history.append(w)
        cost_history.append(cost(w))
    return weight_history, cost_history