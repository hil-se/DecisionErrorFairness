import numpy as np
from data_reader import load_scut
from metrics import Metrics
from pdb import set_trace
from vgg import VGG

class RelativeFairnessTesting():

    def __init__(self):
        self.data, self.protected = load_scut()
        self.features = np.array([pixel for pixel in self.data['pixels']])


    def run(self):
        n = len(self.data)
        train = list(np.random.choice(n, int(n*0.7), replace=True))
        test = list(set(range(n)) - set(train))

        results = []
        base = "P1"
        cols = ["P1", "P2", "P3", "Average"]

        X_train = self.features[train]
        X_test = self.features[test]
        y_train = np.array(self.data[base][train])
        predicts = self.learn(X_train, y_train, X_test)

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

    def learn(self, X, y, X_test):
        # train a model on the training set and use the model to predict on the test set
        model = VGG()
        model.fit(X, y)
        preds = model.predict(X_test)
        return preds


