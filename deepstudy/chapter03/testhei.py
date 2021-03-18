import tensorflow as tf
import numpy as np


a = np.array([[1.0,2.0,3.0],
              [0.0,1.0,2.0],
              [3.0,0.0,1.0]])

b = np.array([[2.0,0.0,1.0],
               [0.0,1.0,2.0],
              [1.0,0.0,3.0]])

print(np.dot(a,b))