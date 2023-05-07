from rft import RelativeFairnessTesting
import pandas as pd

def load():
    exp = RelativeFairnessTesting()
    print(exp.features[0])

def run():
    exp = RelativeFairnessTesting()
    exp.run()


if __name__ == "__main__":
    run()
    # load()
