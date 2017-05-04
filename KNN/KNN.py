import numpy as np

class K_NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y, k):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        self.Xtr = X
        self.ytr = y
        self.k = k

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            dist_sorted = np.argsort(distances)
            classcount = dict()
            for j in xrange(self.k):
                votelabel = self.ytr[dist_sorted[j]]
                classcount[votelabel] = classcount.get(votelabel, 0) + 1
                maxcount = 0
            for key, value in classcount.items():
                if value > maxcount:
                    maxcount = value
                    maxindex = key

            Ypred[i] = maxindex

        return Ypred