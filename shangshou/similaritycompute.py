from numpy import *

def EuclideanDistance(a,b):
    return sqrt((a[0]-b[0])** 2 + (a[1]-b[1])**2)

def ManhattanDistance(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def ChebyshevDistance(a,b):
    return max(abs(a[0] - b[0]),abs(a[1] - b[1]))

def CosineSimilarity(a,b):
    return (a[0]*a[1] + b[0] * b[1]) / (sqrt((a[0] ** 2 + b[0] **2)) * sqrt(a[1] **2 + b[1]**2))

print(EuclideanDistance((1,1),(2,2)))
print(ManhattanDistance((1,1),(2,2)))
print(ChebyshevDistance((1,1),(2,3)))
print(CosineSimilarity((1,1),(1,1)))