from demos import cmd
from experiment import Experiment
import pandas as pd
import numpy as np
from pdb import set_trace

small=0.2
large=0.5

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
    injects = [None, {"sex": small}, {"sex": large}, {"sex": -small}, {"sex": -large}, {"race": small}, {"race": large},
               {"race": -small}, {"race": -large}, {"sex": small, "race": small}, {"sex": -small, "race": -small},
               {"sex": small, "race": -small}, {"sex": -small, "race": small}]
    outputs = [run_inject(data="Adult", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/adult.csv", index=False)

def inject_Heart():
    injects = [None, {"age":small}, {"age":large}, {"age":-small}, {"age":-large}]
    outputs = [run_inject(data="Heart", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/heart.csv", index=False)

def inject_Default():
    injects = [None, {"SEX":small}, {"SEX":large}, {"SEX":-small}, {"SEX":-large}]
    outputs = [run_inject(data="Default", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/default.csv", index=False)

def inject_Bank():
    injects = [None, {"age":small}, {"age":large}, {"age":-small}, {"age":-large}]
    outputs = [run_inject(data="Bank", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/bank.csv", index=False)

def inject_Compas():
    injects = [None, {"sex": small}, {"sex": large}, {"sex": -small}, {"sex": -large}, {"race": small}, {"race": large},
               {"race": -small}, {"race": -large}, {"sex": small, "race": small}, {"sex": -small, "race": -small},
               {"sex": small, "race": -small}, {"sex": -small, "race": small}]
    outputs = [run_inject(data="Compas", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/compas.csv", index=False)

def inject_German():
    injects = [None, {"sex": small}, {"sex": large}, {"sex": -small}, {"sex": -large}, {"age": small}, {"age": large},
               {"age": -small}, {"age": -large}, {"sex": small, "age": small}, {"sex": -small, "age": -small},
               {"sex": small, "age": -small}, {"sex": -small, "age": small}]
    outputs = [run_inject(data="German", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/german.csv", index=False)

def inject_StudentMat():
    injects = [None, {"sex":small}, {"sex":large}, {"sex":-small}, {"sex":-large}]
    outputs = [run_inject(data="StudentMat", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/studentmat.csv", index=False)

def inject_StudentPor():
    injects = [None, {"sex":small}, {"sex":large}, {"sex":-small}, {"sex":-large}]
    outputs = [run_inject(data="StudentPor", inject=inject) for inject in injects]
    df = pd.DataFrame(outputs)
    df.to_csv("../inject_results/studentPor.csv", index=False)

def inject_All():
    inject_Adult()
    inject_Heart()
    inject_Default()
    inject_Bank()
    inject_Compas()
    inject_German()
    inject_StudentMat()
    inject_StudentPor()




if __name__ == "__main__":
    # adult(density_model = 'Neighbor', repeat = 5)
    # print("done")
    eval(cmd())
