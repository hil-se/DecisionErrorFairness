from data_reader import load_scut
from rft import RelativeFairnessTesting
import pandas as pd


def run(base="P1"):
    exp = RelativeFairnessTesting()
    results = exp.run(base)
    df = pd.DataFrame(results)
    df.to_csv("../results/result_"+base+".csv", index=False)

if __name__ == "__main__":
    run("P1")