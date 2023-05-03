from rft import RelativeFairnessTesting
import pandas as pd

def load():
    exp = RelativeFairnessTesting()
    print(exp.features[0])

def run(sex=1, base="P1"):
    exp = RelativeFairnessTesting()
    results = exp.run(base)
    df = pd.DataFrame(results)
    df.to_csv("../results/result"+str(sex)+"_"+base+".csv", index=False)

if __name__ == "__main__":
    run(1, "P1")
    run(0, "P1")
    # load()
