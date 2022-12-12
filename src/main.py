from demos import cmd
from experiment import Experiment
import pandas as pd
import numpy as np
from pdb import set_trace



def run_inject(data="Adult", regressor="Logistic", inject=None, repeat = 30):
    runner = Experiment(data=data, regressor=regressor, inject=inject)
    results = []
    for i in range(repeat):
        result = runner.run()
        results.append(result)
    df = pd.DataFrame(results)
    output = {"Inject": str(inject)}
    for key in df.keys():
        output[key] = "%.2f +/- %.2f" % (np.mean(df[key]), np.std(df[key]))
    print(output)
    return output

def inject_Adult():
    injects = [None, {"sex":0.1}, {"sex":0.2}, {"sex":-0.1}, {"sex":-0.2}, {"race":0.1}, {"race":0.2}, {"race":-0.1}, {"race":-0.2}, {"sex": 0.1, "race":0.1}, {"sex": -0.1, "race":-0.1}, {"sex": 0.1, "race":-0.1}, {"sex": -0.1, "race":0.1}]
    outputs = [run_inject(data="Adult", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/adult.csv", index=False)

if __name__ == "__main__":
    # adult(density_model = 'Neighbor', repeat = 5)
    # print("done")
    eval(cmd())
