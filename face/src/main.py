from rft import RelativeFairnessTesting
import time
import numpy as np

def run():
    np.random.seed(10)
    start = time.time()
    exp = RelativeFairnessTesting(rating_cols = ["P1", "P2", "P3", "Average"])
    exp.run()
    runtime = time.time()-start
    print(runtime)


if __name__ == "__main__":
    run()
