import numpy as np
from metrics import Metrics
import pandas as pd
from bridge import Bridge
from biased_bridge import BiasedBridge
from pdb import set_trace


def run(base="P1"):
    cols = ["P1", "P2", "P3", "Average"]
    protected = ["sex", "race"]
    results = []
    for target in cols:
        data = pd.read_csv("../outputs/"+base+"_"+target+".csv")
        train = data['split']==1
        test = data['split']==0
        # GT on training set
        result = {"Pair": base + "/" + target, "Metric": "GT Train"}
        m = Metrics(data["target"][train], data["base"][train])
        result["MAE"] = m.mae()

        # preds = np.array([1.0 if pred >= 0.5 else 0.0 for pred in data['pred']])
        preds = np.array(data['pred'])

        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][train]), m.RBD(data[A][train]))
        results.append(result)
        # GT on test set
        result = {"Pair": base + "/" + target, "Metric": "GT Test"}
        m = Metrics(data["target"][test], data["base"][test])
        result["MAE"] = m.mae()
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][test]), m.RBD(data[A][test]))
        results.append(result)
        # Prediction on test set
        result = {"Pair": base + "/" + target, "Metric": "Unbiased Bridge"}
        m = Metrics(data["target"][test], preds[test])
        result["MAE"] = m.mae()
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][test]), m.RBD(data[A][test]))
        results.append(result)
        # predict test
        result = {"Pair": base + "/" + target, "Metric": "Biased Bridge"}
        m = BiasedBridge(preds[train] - data["base"][train], preds[test] - data["target"][test])
        result["MAE"] = 0.0
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][train], data[A][test]), m.RBD(data[A][train], data[A][test]))
        results.append(result)
        # predict test bridge
        # result = {"Pair": base + "/" + target, "Metric": "Bridge"}
        # m = Bridge(data, protected)
        # result["MAE"] = 0.0
        # for A in protected:
        #     result[A] = "(%.2f) %.2f" % (
        #     m.RBT(A), m.RBD(A))
        # results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../results/output_result_" + base + ".csv", index=False)

if __name__ == "__main__":
    cols = ["P1", "P2", "P3", "Average"]
    for base in cols:
        run(base=base)
