import numpy as np
from data_reader import load_scut
from clf_metrics import Clf_Metrics
from vgg_pre import VGG_Pre
from preprocessor import *
import time
from pdb import set_trace

class exp():

    def __init__(self, rating_cols = ["P1", "P2", "P3", "Average"]):
        self.rating_cols = rating_cols
        self.data, self.protected = load_scut(rating_cols = rating_cols)
        self.features = np.array([pixel for pixel in self.data['pixels']])/255.0

    def run(self, base = "Average", treatments = ["None"]):
        n = len(self.data)
        test = list(np.random.choice(n, int(n * 0.4), replace=False))
        train = list(set(range(n)) - set(test))
        val = list(np.random.choice(test, int(n * 0.2), replace=False))
        test = list(set(test) - set(val))

        X_train = self.features[train]
        X_val = self.features[val]

        y_train = np.array(self.data[base][train])
        y_val = np.array(self.data[base][val])

        data_train = self.data.loc[train]
        data_train.index = range(len(data_train))
        data_val = self.data.loc[val]
        data_val.index = range(len(data_val))
        data_test = self.data.loc[test]
        data_test.index = range(len(data_test))
        metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime"]
        columns = ["Treatment"] + metrics
        test_result = {column: [] for column in columns}
        for treatment in treatments:
            if treatment=="Reweighing":
                sample_weight = Reweighing(data_train, y_train, self.protected)
                val_sample_weights = Reweighing(data_val, y_val, self.protected)
            elif treatment=="FairBalance":
                sample_weight = FairBalance(data_train, y_train, self.protected)
                val_sample_weights = FairBalance(data_val, y_val, self.protected)
            elif treatment=="FairBalanceVariant":
                sample_weight = FairBalanceVariant(data_train, y_train, self.protected)
                val_sample_weights = FairBalanceVariant(data_val, y_val, self.protected)
            else:
                sample_weight = None
                val_sample_weights = None

            start_time = time.time()
            self.learn(X_train, y_train, X_val, y_val, sample_weight, val_sample_weights)
            runtime = time.time() - start_time
            preds = self.model.predict(self.features)
            decs = self.model.decision_function(self.features).flatten()


            m_test = Clf_Metrics(data_test, np.array(self.data[base][test]), preds[test], decs[test], self.protected)
            test_result["Treatment"].append(treatment)
            test_result["Accuracy"].append("%.2f" % m_test.accuracy())
            test_result["AUC"].append("%.2f" % m_test.auc())
            test_result["mEOD"].append("%.2f" % m_test.eod())
            test_result["mAOD"].append("%.2f" % m_test.aod())
            test_result["smEOD"].append("%.2f" % m_test.seod())
            test_result["smAOD"].append("%.2f" % m_test.saod())
            test_result["Runtime"].append("%.2f" % runtime)
        return test_result

    def run2(self, base = "Average", treatments = ["None"]):
        n = len(self.data)
        test = list(np.random.choice(n, int(n * 0.3), replace=False))
        train_all = list(set(range(n)) - set(test))
        val = list(np.random.choice(train_all, int(n * 0.2), replace=False))
        train = list(set(train_all) - set(val))
        train_all = train + val

        X_train = self.features[train]
        X_val = self.features[val]

        y_train = np.array(self.data[base][train])
        y_val = np.array(self.data[base][val])

        data_train = self.data.loc[train_all]
        data_train.index = range(len(data_train))
        data_test = self.data.loc[test]
        data_test.index = range(len(data_test))
        metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime"]
        columns = ["Treatment"] + metrics
        test_result = {column: [] for column in columns}
        for treatment in treatments:
            if treatment=="Reweighing":
                weight = Reweighing(data_train, np.array(self.data[base][train_all]), self.protected)
                sample_weight = weight[:len(train)]
                val_sample_weights = weight[len(train):]
            elif treatment=="FairBalance":
                weight = FairBalance(data_train, np.array(self.data[base][train_all]), self.protected)
                sample_weight = weight[:len(train)]
                val_sample_weights = weight[len(train):]
            elif treatment=="FairBalanceVariant":
                weight = FairBalanceVariant(data_train, np.array(self.data[base][train_all]), self.protected)
                sample_weight = weight[:len(train)]
                val_sample_weights = weight[len(train):]
            else:
                sample_weight = None
                val_sample_weights = None

            start_time = time.time()
            self.learn(X_train, y_train, X_val, y_val, sample_weight, val_sample_weights)
            runtime = time.time() - start_time
            preds = self.model.predict(self.features)
            decs = self.model.decision_function(self.features).flatten()


            m_test = Clf_Metrics(data_test, np.array(self.data[base][test]), preds[test], decs[test], self.protected)
            test_result["Treatment"].append(treatment)
            test_result["Accuracy"].append("%.2f" % m_test.accuracy())
            test_result["AUC"].append("%.2f" % m_test.auc())
            test_result["mEOD"].append("%.2f" % m_test.eod())
            test_result["mAOD"].append("%.2f" % m_test.aod())
            test_result["smEOD"].append("%.2f" % m_test.seod())
            test_result["smAOD"].append("%.2f" % m_test.saod())
            test_result["Runtime"].append("%.2f" % runtime)
        return test_result

    def learn(self, X, y, X_val, y_val, sample_weight=None, val_sample_weights=None):
        # train a model on the training set and use the model to predict on the test set
        # model = VGG()
        # self.model = VGG_Pre(saved_model = "./checkpoint/attractiveness.keras")
        self.model = VGG_Pre()
        self.model.fit(X, y, X_val, y_val, sample_weight=sample_weight, val_sample_weights=val_sample_weights)