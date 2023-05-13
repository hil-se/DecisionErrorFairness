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

    def stats(self, group_train, group_test):
        mu_train, var_train = self.norm_stats(self.delta_train[group_train])
        mu_test, var_test = self.norm_stats(self.delta_test[group_test])
        mu = mu_test - mu_train
        var = var_test + var_train
        return mu, var

    def ERBT(self, s_train, s_test):
        if len(np.unique(s_train)) == 2 and len(np.unique(s_test)) == 2:
            group0_train = np.where(np.array(s_train) == 0)[0]
            group0_test = np.where(np.array(s_test) == 0)[0]
            group1_train = np.where(np.array(s_train) == 1)[0]
            group1_test = np.where(np.array(s_test) == 1)[0]
            mu0, var0 = self.stats(group0_train, group0_test)
            mu1, var1 = self.stats(group1_train, group1_test)
            erbt = (mu1 - mu0) / np.sqrt(var1 / (len(group1_test)+len(group1_train)) + var0 / (len(group0_test)+len(group0_train)))
        else:
            # bias_diff = 0.0
            # n = 0
            # delta = self.delta_train - self.delta_test
            # for i in range(len(s)):
            #     for j in range(len(s)):
            #         if np.array(s)[i] - np.array(s)[j] > 0:
            #             n += 1
            #             bias_diff += delta[i]-delta[j]
            # bias_diff = bias_diff / n
            # mu_train, var_train = self.norm_stats(self.delta_train)
            # mu_test, var_test = self.norm_stats(self.delta_test)
            # var = var_train + var_test
            # if var:
            #     bias_diff = bias_diff / np.sqrt(var)
            # else:
            #     bias_diff = 0.0
            erbt = 0.0
        return erbt

    def ERBD(self, s_train, s_test):
        group0_train = np.where(np.array(s_train) == 0)[0]
        group0_test = np.where(np.array(s_test) == 0)[0]
        group1_train = np.where(np.array(s_train) == 1)[0]
        group1_test = np.where(np.array(s_test) == 1)[0]
        mu0, var0 = self.stats(group0_train, group0_test)
        mu1, var1 = self.stats(group1_train, group1_test)
        erbd = (mu1 - mu0) / np.sqrt(
            var1 * (len(group1_test) + len(group1_train) - 1) + var0 * (len(group0_test) + len(group0_train) - 1)/ (len(group1_test) + len(group1_train) + len(group0_test) + len(group0_train)))
        return erbd
