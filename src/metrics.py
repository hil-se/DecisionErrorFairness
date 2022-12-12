import sklearn.metrics
import numpy as np
from pdb import set_trace

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def r2(self):
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def pairwise(self):
        t=tp=fn=0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if self.y[i] - self.y[j]>0:
                    t+=1
                    if self.y_pred[i] > self.y_pred[j]:
                        tp+=1
                    elif self.y_pred[i] < self.y_pred[j]:
                        fn+=1
        return tp/t, fn/t

    def AOD(self, s):
        # s is an array of numerical values of a sensitive attribute
        t= n= tp= fp= tn= fn = 0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i]-s[j]>0:
                    if self.y[i]-self.y[j]>0:
                        t+=1
                        if self.y_pred[i]>self.y_pred[j]:
                            tp+=1
                        if self.y_pred[i]<self.y_pred[j]:
                            fn+=1
                    elif self.y[j]-self.y[i]>0:
                        n+=1
                        if self.y_pred[i]>self.y_pred[j]:
                            fp+=1
                        elif self.y_pred[i]<self.y_pred[j]:
                            tn+=1

        tpr = tp / t
        tnr = tn / n
        fpr = fp / n
        fnr = fn / t
        aod = (tpr+fpr-tnr-fnr)/2
        return aod

    def AODc(self, s):
        # s is an array of numerical values of a sensitive attribute
        t= n= tp= fp = 0.0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i]-s[j]>0:
                    y_diff = self.y[i] - self.y[j]
                    y_pred_diff = self.y_pred[i] - self.y_pred[j]
                    if y_diff>0:
                        t+=y_diff
                        tp+=y_pred_diff
                    elif y_diff<0:
                        n+=y_diff
                        fp+=y_pred_diff

        aod = (tp/t-fp/n)/2
        return aod

    def AODc2(self, s):
        # s is an array of numerical values of a sensitive attribute
        t= n= tp= fp = 0.0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i]-s[j]>0:
                    y_diff = self.y[i] - self.y[j]
                    y_pred_diff = self.y_pred[i] - self.y_pred[j]
                    if y_diff>0:
                        t+=y_diff**2
                        tp+=y_diff*y_pred_diff
                    elif y_diff<0:
                        n+=y_diff**2
                        fp+=y_diff*y_pred_diff

        aod = (tp/t-fp/n)/2
        return aod

    def BiasDiff(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            group0 = max(np.unique(s))
            group1 = min(np.unique(s))
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[group0] = error[np.where(s==group0)[0]]
            bias[group1] = error[np.where(s==group1)[0]]
            bias_diff = np.mean(bias[group0]) - np.mean(bias[group1])
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                error_i = self.y_pred[i] - self.y[i]
                for j in range(len(self.y)):
                    error_j = self.y_pred[j] - self.y[j]
                    if s[i] - s[j] > 0:
                        n += 1
                        bias_diff += error_i - error_j
            bias_diff = bias_diff / n
        sigma = np.std(self.y_pred - self.y)
        if sigma:
            bias_diff = bias_diff / sigma
        else:
            bias_diff = 0.0
        return bias_diff

    def NullHypo(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            group0 = max(np.unique(s))
            group1 = min(np.unique(s))
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[group0] = error[np.where(s == group0)[0]]
            bias[group1] = error[np.where(s == group1)[0]]
            bias_diff = np.mean(bias[group0]) - np.mean(bias[group1])
            sigma = np.sqrt(np.std(bias[group0])**2/len(bias[group0]) + np.std(bias[group1])**2/len(bias[group1]))
            if sigma:
                bias_diff = bias_diff / sigma
            else:
                bias_diff = 0.0
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                error_i = self.y_pred[i] - self.y[i]
                for j in range(len(self.y)):
                    error_j = self.y_pred[j] - self.y[j]
                    if s[i] - s[j] > 0:
                        n += 1
                        bias_diff += error_i - error_j
            bias_diff = bias_diff / n
            sigma = np.std(self.y_pred - self.y)
            if sigma:
                bias_diff = bias_diff * np.sqrt(len(s)) / sigma
            else:
                bias_diff = 0.0
        # bias_diff = scipy.stats.norm.sf(abs(bias_diff))
        return bias_diff







