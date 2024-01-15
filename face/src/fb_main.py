from fb_exp import exp
import pandas as pd
import time

def run():
    treatments = ["None", "Reweighing", "FairBalance", "FairBalanceVariant"]
    treatments = ["None"]
    cols = ["P1", "P2", "P3", "Average"]
    cols = ["Average"]
    runner = exp(rating_cols = cols)
    for base in cols:
        metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime"]
        columns = ["Treatment"] + metrics
        test_result = {column: [] for column in columns}
        for treatment in treatments:
            start = time.time()
            m_test = runner.run(base=base, treatment=treatment)
            runtime = time.time()-start
            test_result["Treatment"].append(treatment)
            test_result["Accuracy"].append("%.2f" %m_test.accuracy())
            test_result["AUC"].append("%.2f" %m_test.auc())
            test_result["mEOD"].append("%.2f" %m_test.eod())
            test_result["mAOD"].append("%.2f" %m_test.aod())
            test_result["smEOD"].append("%.2f" %m_test.seod())
            test_result["smAOD"].append("%.2f" %m_test.saod())
            test_result["Runtime"].append("%.2f" %runtime)
        df_test = pd.DataFrame(test_result, columns=columns)
        df_test.to_csv("../results/fb_" + base + ".csv", index=False)




if __name__ == "__main__":
    run()