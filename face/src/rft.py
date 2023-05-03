import numpy as np
from data_reader import load_scut
from metrics import Metrics
from pdb import set_trace
from vgg import VGG
from vgg_pre import VGG_Pre

class RelativeFairnessTesting():

    def __init__(self, sex = 1):
        data, self.protected = load_scut()
        self.data = data[self.data["sex"]==sex]
        self.data.index = range(len(self.data))
        self.features = np.array([pixel for pixel in self.data['pixels']])/255.0


    def run(self, base="P1"):
        n = len(self.data)
        # train = list(np.random.choice(n, int(n*0.7), replace=True))
        # test = list(set(range(n)) - set(train))
        train, test = self.train_test_split(test_size=0.3, base=base)

        results = []
        cols = ["P1", "P2", "P3", "Average"]

        X_train = self.features[train]
        X_test = self.features[test]
        y_train = np.array(self.data[base][train])
        predicts = self.learn(X_train, y_train, X_test, base=base)

        for target in cols:
            # GT on training set
            result = {"Pair": base+"/"+target, "Metric": "GT Train"}
            m = Metrics(self.data[target][train], self.data[base][train])
            result["Accuracy"] = 1.0 - m.mae()
            for A in self.protected:
                result[A+": "+"CBT"] = "%.2f" %m.CBT(self.data[A][train])
                result[A+": "+"CBD"] = "%.2f" %m.CBD(self.data[A][train])
            results.append(result)
            # GT on test set
            result = {"Pair": base+"/"+target, "Metric": "GT Test"}
            m = Metrics(self.data[target][test], self.data[base][test])
            result["Accuracy"] = 1.0 - m.mae()
            for A in self.protected:
                result[A + ": " + "CBT"] = "%.2f" %m.CBT(self.data[A][test])
                result[A + ": " + "CBD"] = "%.2f" %m.CBD(self.data[A][test])
            results.append(result)
            # Prediction on test set
            result = {"Pair": base + "/" + target, "Metric": "Pred Test"}
            m = Metrics(self.data[target][test], predicts)
            result["Accuracy"] = 1.0 - m.mae()
            for A in self.protected:
                result[A + ": " + "CBT"] = "%.2f" %m.CBT(self.data[A][test])
                result[A + ": " + "CBD"] = "%.2f" %m.CBD(self.data[A][test])
            results.append(result)

        return results

    def train_test_split(self, test_size=0.3, base = "Average"):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.data)):
            key = tuple([self.data[a][i] for a in self.protected] + [self.data[base][i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key])*test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)
        return train, test

    def learn(self, X, y, X_test, base = "P1"):
        # train a model on the training set and use the model to predict on the test set
        # model = VGG()
        model = VGG_Pre()
        model.fit(X, y, base=base)
        # preds = model.predict(X_test)
        preds = model.decision_function(X_test).flatten()
        print(np.unique(preds))
        return preds


