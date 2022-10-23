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
def linear_regression(x,y):
    

