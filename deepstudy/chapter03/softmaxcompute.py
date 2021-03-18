import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    exp_sum = np.sum(exp_x)
    return exp_x / exp_sum


data = np.array([0.3,2.9,4.0])
exp_data = np.exp(data)
print(exp_data)
data_exp_sum = np.sum(exp_data)
data_result = exp_data / data_exp_sum
print(data_result)
print(softmax(data))