import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

data = np.array([-33,-4,0,20])

print(relu(data))
# data = np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10])
# print(sigmoid(data))
# x=[2,8]
# y=[1,28]
# plt.axis([-10,10,-1,2])
# plt.plot(data,sigmoid(data))
# plt.show()
