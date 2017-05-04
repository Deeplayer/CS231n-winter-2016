import numpy as np
import matplotlib.pyplot as plt

x = np.zeros((4,3,5,5))
print x.shape
x[2,1,:,:] = np.ones((5,5))
y = np.max(x,axis=1)
print y
