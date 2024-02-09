import numpy as np
from data_reader import load_scut
from clf_metrics import Clf_Metrics
from vgg_pre import VGG_Pre
from preprocessor import *
import time
import tensorflow as tf
from random import shuffle
from pdb import set_trace

class exp():

    def __init__(self, rating_cols = ["P1", "P2", "P3", "Average"]):
        self.rating_cols = rating_cols
        self.data, self.protected = load_scut(rating_cols = rating_cols)
        self.features = np.array([pixel for pixel in self.data['pixels']])/255.0

    def run(self, base = "Average", treatments = ["None"]):
        # n = len(self.data)
        # test = list(np.random.choice(n, int(n * 0.3), replace=False))
        # train = list(set(range(n)) - set(test))
        # val = list(np.random.choice(train, int(n * 0.2), replace=False))
        # train = list(set(train) - set(val))


        train, test, val = self.train_test_split(base, test_size=0.3, val_size=0.2)

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

        # metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime"]
        metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime", "bce", "bce_train", "bce_train_weight", "bce_val", "bce_val_weight", "smAOD_train", "smAOD_val"]
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
            # X_val = []
            # y_val = []
            # val_sample_weights = None

            start_time = time.time()
            train_data = BalancedSequence(X_train, y_train, data_train, self.protected, batch_size=150,
                                          sample_weight=sample_weight)
            val_data = BalancedSequence(X_val, y_val, data_val, self.protected, batch_size=150,
                                          sample_weight=val_sample_weights)
            self.learn2(train_data, val_data)
            runtime = time.time() - start_time
            preds = self.model.predict(self.features)
            decs = self.model.decision_function(self.features).flatten()


            m_test = Clf_Metrics(data_test, np.array(self.data[base][test]), preds[test], decs[test], self.protected)
            test_result["Treatment"].append(treatment)
            test_result["Accuracy"].append(m_test.accuracy())
            test_result["AUC"].append(m_test.auc())
            test_result["mEOD"].append(m_test.eod())
            test_result["mAOD"].append(m_test.aod())
            test_result["smEOD"].append(m_test.seod())
            test_result["smAOD"].append(m_test.saod())
            test_result["Runtime"].append(runtime)
            test_result["bce"].append(m_test.bce())

            m_train = Clf_Metrics(data_train, np.array(self.data[base][train]), preds[train], decs[train], self.protected)
            test_result["smAOD_train"].append(m_train.saod())
            test_result["bce_train"].append(m_train.bce())
            test_result["bce_train_weight"].append(m_train.bce(sample_weight=sample_weight))

            m_val = Clf_Metrics(data_val, np.array(self.data[base][val]), preds[val], decs[val],
                                  self.protected)
            test_result["smAOD_val"].append(m_val.saod())
            test_result["bce_val"].append(m_val.bce())
            test_result["bce_val_weight"].append(m_val.bce(sample_weight=val_sample_weights))


        return test_result

    # def run(self, base = "Average", treatments = ["None"]):
    #     # n = len(self.data)
    #     # test = list(np.random.choice(n, int(n * 0.3), replace=False))
    #     # train = list(set(range(n)) - set(test))
    #     # val = list(np.random.choice(train, int(n * 0.2), replace=False))
    #     # train = list(set(train) - set(val))
    #
    #     train, test, val = self.train_test_split(base, test_size=0.3, val_size=0.2)
    #
    #     X_train = self.features[train]
    #     X_val = self.features[val]
    #
    #     y_train = np.array(self.data[base][train])
    #     y_val = np.array(self.data[base][val])
    #
    #     data_train = self.data.loc[train]
    #     data_train.index = range(len(data_train))
    #     data_val = self.data.loc[val]
    #     data_val.index = range(len(data_val))
    #     data_test = self.data.loc[test]
    #     data_test.index = range(len(data_test))
    #
    #     # metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime"]
    #     metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime", "bce", "bce_train",
    #                "bce_train_weight", "bce_val", "bce_val_weight", "smAOD_train", "smAOD_val"]
    #     columns = ["Treatment"] + metrics
    #     test_result = {column: [] for column in columns}
    #     for treatment in treatments:
    #         if treatment == "Reweighing":
    #             sample_weight = Reweighing(data_train, y_train, self.protected)
    #             val_sample_weights = Reweighing(data_val, y_val, self.protected)
    #         elif treatment == "FairBalance":
    #             sample_weight = FairBalance(data_train, y_train, self.protected)
    #             val_sample_weights = FairBalance(data_val, y_val, self.protected)
    #         elif treatment == "FairBalanceVariant":
    #             sample_weight = FairBalanceVariant(data_train, y_train, self.protected)
    #             val_sample_weights = FairBalanceVariant(data_val, y_val, self.protected)
    #         else:
    #             sample_weight = None
    #             val_sample_weights = None
    #         # X_val = []
    #         # y_val = []
    #         # val_sample_weights = None
    #
    #         start_time = time.time()
    #         self.learn(X_train, y_train, X_val, y_val, sample_weight, val_sample_weights)
    #         runtime = time.time() - start_time
    #         preds = self.model.predict(self.features)
    #         decs = self.model.decision_function(self.features).flatten()
    #
    #         m_test = Clf_Metrics(data_test, np.array(self.data[base][test]), preds[test], decs[test], self.protected)
    #         test_result["Treatment"].append(treatment)
    #         test_result["Accuracy"].append(m_test.accuracy())
    #         test_result["AUC"].append(m_test.auc())
    #         test_result["mEOD"].append(m_test.eod())
    #         test_result["mAOD"].append(m_test.aod())
    #         test_result["smEOD"].append(m_test.seod())
    #         test_result["smAOD"].append(m_test.saod())
    #         test_result["Runtime"].append(runtime)
    #         test_result["bce"].append(m_test.bce())
    #
    #         m_train = Clf_Metrics(data_train, np.array(self.data[base][train]), preds[train], decs[train],
    #                               self.protected)
    #         test_result["smAOD_train"].append(m_train.saod())
    #         test_result["bce_train"].append(m_train.bce())
    #         test_result["bce_train_weight"].append(m_train.bce(sample_weight=sample_weight))
    #
    #         m_val = Clf_Metrics(data_val, np.array(self.data[base][val]), preds[val], decs[val],
    #                             self.protected)
    #         test_result["smAOD_val"].append(m_val.saod())
    #         test_result["bce_val"].append(m_val.bce())
    #         test_result["bce_val_weight"].append(m_val.bce(sample_weight=val_sample_weights))
    #     return test_result


    def train_test_split(self, base, test_size=0.3, val_size=0.2):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.data)):
            key = tuple([self.data[a][i] for a in self.protected] + [self.data[base][i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        val = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key]) * test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            training2 = list(np.random.choice(training, int(len(groups[key]) * (1.0 - val_size - test_size)), replace=False))
            valing = list(set(training) - set(training2))
            test.extend(testing)
            train.extend(training2)
            val.extend(valing)
        return train, test, val


    def learn(self, X, y, X_val, y_val, sample_weight=None, val_sample_weights=None):
        # train a model on the training set and use the model to predict on the test set
        # model = VGG()
        # self.model = VGG_Pre(saved_model = "./checkpoint/attractiveness.keras")
        self.model = VGG_Pre()
        self.model.fit(X, y, X_val, y_val, sample_weight=sample_weight, val_sample_weights=val_sample_weights)

    def learn2(self, train_data, val_data):
        # train a model on the training set and use the model to predict on the test set
        self.model = VGG_Pre()
        self.model.fit2(train_data, val_data)

class BalancedSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, data, protected, batch_size=100, sample_weight=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.data = data
        self.protected = protected
        self.groups = {}
        for i in range(len(self.data)):
            key = tuple([self.data[a][i] for a in self.protected] + [self.y[i]])
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(i)
        self.batches = {}
        size = 0
        for key in self.groups:
            self.batches[key] = int(float(len(self.groups[key]))*self.batch_size/len(self.y))
            size += self.batches[key]
        self.batch_size = size

    def __len__(self):
        return np.ceil(len(self.y) / self.batch_size)

    def on_epoch_end(self):
        for key in self.groups:
            shuffle(self.groups[key])

    def __getitem__(self, idx):
        samples = []
        for key in self.groups:
            samples.extend(self.groups[key][idx * self.batches[key]:(idx+1) * self.batches[key]])
        if self.sample_weight is None:
            sample_weight = None
        else:
            sample_weight = self.sample_weight[samples]
        return self.x[samples], self.y[samples], sample_weight