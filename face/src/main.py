from data_reader import load_scut
from rft import RelativeFairnessTesting
import pandas as pd
from pdb import set_trace

def load():
    data, A = load_scut()
    print(data)

def run():
    exp = RelativeFairnessTesting()
    results = exp.run()
    df = pd.DataFrame(results)
    df.to_csv("../results/result.csv", index=False)

if __name__ == "__main__":
    run()