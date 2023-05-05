from rft import RelativeFairnessTesting
import pandas as pd

def load():
    exp = RelativeFairnessTesting()
    print(exp.features[0])

def run(base="P1"):
    exp = RelativeFairnessTesting()
    results = exp.run(base)
    df = pd.DataFrame(results)
    df.to_csv("../results/result_"+base+".csv", index=False)

if __name__ == "__main__":
    run("P1")
    run("P2")
    run("P3")
    run("Average")
    # load()
