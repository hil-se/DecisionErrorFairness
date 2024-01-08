import numpy as np
from data_reader import load_scut
from clf_metrics import Clf_Metrics
from vgg_pre import VGG_Pre

class exp():

    def __init__(self, rating_cols = ["P1", "P2", "P3", "Average"]):
        self.rating_cols = rating_cols
        self.data, self.protected = load_scut(rating_cols = rating_cols)
        self.features = np.array([pixel for pixel in self.data['pixels']])/255.0

    def run(self, base = "Average"):
        n = len(self.data)
        test = list(np.random.choice(n, int(n * 0.4), replace=False))
        train = list(set(range(n)) - set(test))
        val = list(np.random.choice(test, int(n * 0.2), replace=False))
        test = list(set(test) - set(val))
        training = list(set(train) | set(val))

        X_train = self.features[train]
        X_val = self.features[val]
        X_test = self.features[test]

        y_train = np.array(self.data[base][train])
        y_val = np.array(self.data[base][val])
        self.learn(X_train, y_train, X_val, y_val)
        preds = self.model.predict(self.features)
        decs = self.model.decision_function(self.features).flatten()
        data_test = self.data.loc[test]
        data_test.index = range(len(data_test))

        m = Clf_Metrics(data_test, np.array(self.data[base][test]), preds[test], decs[test], self.protected)
        return m

    def learn(self, X, y, X_val, y_val):
        # train a model on the training set and use the model to predict on the test set
        # model = VGG()
        self.model = VGG_Pre()
        self.model.fit(X, y, X_val, y_val)