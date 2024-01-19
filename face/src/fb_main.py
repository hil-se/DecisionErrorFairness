from fb_exp import exp
import pandas as pd
import numpy as np
from demos import cmd

def run(base="Average", repeats = 10):
    treatments = ["None", "Reweighing", "FairBalance", "FairBalanceVariant"]
    # treatments = ["None"]
    runner = exp(rating_cols = [base])
    result = None
    for _ in range(repeats):
        test_result = runner.run(base=base, treatments=treatments)
        if result is None:
            result = {key: test_result[key] if key == "Treatment" else [[value] for value in test_result[key]]  for key in test_result}
            continue
        for key in test_result:
            if key == "Treatment":
                continue
            for i, value in enumerate(test_result[key]):
                result[key][i].append(value)
    for key in result:
        if key == "Treatment":
            continue
        result[key] = [
            "%.2f (%.2f)" % (np.median(l), np.quantile(result[key], 0.75) - np.quantile(result[key], 0.25)) for l in
            result[key]]
    df_test = pd.DataFrame(result)
    df_test.to_csv("../results/fb_" + base + ".csv", index=False)

def run2():
    treatments = ["None", "Reweighing", "FairBalance", "FairBalanceVariant"]
    # treatments = ["None"]
    cols = ["P1", "P2", "P3", "Average"]
    # cols = ["Average"]
    runner = exp(rating_cols = cols)
    for base in cols:
        test_result = runner.run2(base=base, treatments=treatments)
        df_test = pd.DataFrame(test_result)
        df_test.to_csv("../results/fb2_" + base + ".csv", index=False)

if __name__ == "__main__":
    eval(cmd())