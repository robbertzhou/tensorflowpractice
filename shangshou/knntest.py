import numpy as np


class KNN:
    def __init__(self,k):
        self.K = k

    def creaetData(self):
        features = np.array([[180,76],[158,43],[176,78],[161,49]])
        labels = np.array(['男','女','男','女'])
        return features,labels

    def Normalization(self,data):
        maxs = np.max(data,axis=0)
        mins = np.min(data,axis=0)
        new_data = (data - mins) /(maxs - mins)
        return new_data,maxs,mins



if __name__ == "__main__":
    pass
