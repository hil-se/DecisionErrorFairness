from fb_exp import exp
import pandas as pd

def run():
    treatments = ["None", "Reweighing", "FairBalance", "FairBalanceVariant"]
    # treatments = ["None"]
    cols = ["P1", "P2", "P3", "Average"]
    # cols = ["Average"]
    runner = exp(rating_cols = cols)
    for base in cols:
        test_result = runner.run(base=base, treatments=treatments)
        df_test = pd.DataFrame(test_result)
        df_test.to_csv("../results/fb_" + base + ".csv", index=False)


if __name__ == "__main__":
    run()