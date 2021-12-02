import os
import numpy as np 
from scipy.special import eval_genlaguerre as L 

# function definitions 
def kld(mu, tau, d):
    # no need to include z, since we run gradient descent...
    return -tau*np.sqrt(np.pi/2)*L(1/2, d/2 -1, -(mu**2)/2) + (mu**2)/2

def kld_min(tau, d):
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    dx = 5e-3

    # inital guess (very close to optimal value)
    x = np.sqrt(max(tau**2 - d, 0))

    # run gradient descent (kld is convex)
    for step in steps:
        for i in range(10000):
            y1 = kld(x-dx/2, tau, d)
            y2 = kld(x+dx/2, tau, d)

            grad = (y2-y1)/dx
            x -= grad*step

    return x
