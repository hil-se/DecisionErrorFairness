from rft import RelativeFairnessTesting
import time

def load():
    exp = RelativeFairnessTesting()
    print(exp.features[0])

def run():
    start = time.time()
    exp = RelativeFairnessTesting()
    exp.run()
    runtime = time.time()-start
    print(runtime)


if __name__ == "__main__":
    run()
