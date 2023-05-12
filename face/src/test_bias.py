import numpy as np

class TestBias:
    def __init__(self, delta_train, delta_test):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.delta_train = delta_train
        self.delta_test = delta_test

    def norm_stats(self, x):
        mu = np.mean(x)
        var = np.var(x)
        return mu, var

    def stats(self, group):
        mu_train, var_train = self.norm_stats(self.delta_train[group])
        mu_test, var_test = self.norm_stats(self.delta_test[group])
        mu = mu_test - mu_train
        var = var_test + var_train
        return mu, var

    def ERBT(self, s):
        group0 = np.where(np.array(s) == 0)[0]
        group1 = np.where(np.array(s) == 1)[0]
        mu0, var0 = self.stats(group0)
        mu1, var1 = self.stats(group1)
        return (mu1-mu0)/np.sqrt(var1/len(group1)+var0/len(group0))

    def ERBD(self, s):
        group0 = np.where(np.array(s) == 0)[0]
        group1 = np.where(np.array(s) == 1)[0]
        mu0, var0 = self.stats(group0)
        mu1, var1 = self.stats(group1)
        return (mu1-mu0)/np.sqrt((var1*(len(group1)-1)+var0*(len(group0)-1))/(len(group1)+len(group0)))
