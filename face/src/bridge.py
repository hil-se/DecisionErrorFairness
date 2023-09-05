import numpy as np
from scipy.stats import t

class Bridge:
    def __init__(self, data, protected):
        self.data = data
        self.protected = protected

    def RBD(self, A):
        groups = {}
        IDs = []
        for i in range(len(self.data)):
            groupID = tuple(self.data[self.protected+["split"]].loc[i])
            if groupID not in groups:
                groups[groupID] = []
            ID = groupID[:-1]
            if ID not in IDs:
                IDs.append(ID)
            groups[groupID].append(i)
        sums = {}
        for ID in IDs:
            delta_test = self.data["pred"][groups[ID + tuple([0])]] - self.data["target"][groups[ID + tuple([0])]]
            delta_train = self.data["pred"][groups[ID + tuple([1])]] - self.data["base"][groups[ID + tuple([1])]]
            sums[ID] = {"mean": np.mean(delta_test) - np.mean(delta_train), "var": np.var(delta_test, ddof=1) + np.var(delta_train, ddof=1), "n": len(delta_test)}
        group0 = []
        group1 = []
        index = self.protected.index(A)
        for ID in IDs:
            if ID[index]==0:
                group0.append(ID)
            else:
                group1.append(ID)
        mean0 = 0
        var0 = 0
        dof0 = 0
        num0 = 0
        for ID in group0:
            mean0 += sums[ID]["mean"] * sums[ID]["n"]
            var0 += sums[ID]["var"] * (sums[ID]["n"]-1)
            dof0 += sums[ID]["n"]-1
            num0 += sums[ID]["n"]
        mean0 = mean0/num0
        mean1 = 0
        var1 = 0
        dof1 = 0
        num1 = 0
        for ID in group1:
            mean1 += sums[ID]["mean"] * sums[ID]["n"]
            var1 += sums[ID]["var"] * (sums[ID]["n"] - 1)
            dof1 += sums[ID]["n"] - 1
            num1 += sums[ID]["n"]
        mean1 = mean1/num1
        erbd = (mean1 - mean0) / (np.sqrt((var0+var1)/(dof0+dof1)))

        return erbd

    def RBT(self, A):
        groups = {}
        IDs = []
        for i in range(len(self.data)):
            groupID = tuple(self.data[self.protected+["split"]].loc[i])
            if groupID not in groups:
                groups[groupID] = []
            ID = groupID[:-1]
            if ID not in IDs:
                IDs.append(ID)
            groups[groupID].append(i)
        sums = {}
        for ID in IDs:
            delta_test = self.data["pred"][groups[ID+tuple([0])]] - self.data["target"][groups[ID+tuple([0])]]
            delta_train = self.data["pred"][groups[ID + tuple([1])]] - self.data["base"][groups[ID + tuple([1])]]
            sums[ID] = {"mean": np.mean(delta_test) - np.mean(delta_train), "var": np.var(delta_test, ddof=1) + np.var(delta_train, ddof=1), "n": len(delta_test)}
        group0 = []
        group1 = []
        index = self.protected.index(A)
        for ID in IDs:
            if ID[index]==0:
                group0.append(ID)
            else:
                group1.append(ID)
        mean0 = 0
        var0 = 0
        dof0 = 0
        num0 = 0
        for ID in group0:
            mean0 += sums[ID]["mean"] * sums[ID]["n"]
            var0 += sums[ID]["var"] / (sums[ID]["n"])
            dof0 += (sums[ID]["var"] / (sums[ID]["n"]))**2/(sums[ID]["n"]-1)
            num0 += sums[ID]["n"]
        mean0 = mean0 / num0
        mean1 = 0
        var1 = 0
        dof1 = 0
        num1 = 0
        for ID in group1:
            mean1 += sums[ID]["mean"] * sums[ID]["n"]
            var1 += sums[ID]["var"] / (sums[ID]["n"])
            dof1 += (sums[ID]["var"] / (sums[ID]["n"]))**2/(sums[ID]["n"]-1)
            num1 += sums[ID]["n"]
        mean1 = mean1 / num1
        erbt = (mean1 - mean0) / (np.sqrt((var0+var1)))
        dof = round((var0+var1)**2/(dof0+dof1))
        p = t.sf(np.abs(erbt), dof)
        return p
