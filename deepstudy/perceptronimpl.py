#感知机实现

import numpy as np

def AND(x,y):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x* w1 + y*w2
    if tmp > theta:
        return 1
    else:
        return 0

def np_and(x1,x2):
    w = np.array([0.5,0.5])
    data = np.array([x1,x2])
    b = -0.7
    tmp = np.sum(w*data) + b
    if tmp > 0.7:
        return 1
    else:
        return 0

print(AND(1,0))

x = np.array([0.0,1.0]) #输入
w = np.array([0.5,0.5]) #权重
xw = w * x
b = -0.7
xw_sum = np.sum(xw)
b_sum = xw_sum + b
print(xw_sum)
print(b_sum)
#print(np.sum(xw))
