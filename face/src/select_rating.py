import pandas as pd
from pdb import set_trace

def select_ratings():
    df = pd.read_csv("../data/All_Ratings.csv")
    selected = {"Filename":[]}
    rater = 0
    for i, x in enumerate(df["Rater"]):
        if x != rater:
            selected["P"+str(x)] = []
            rater = x
        if x == 1:
            selected["Filename"].append(df["Filename"][i])
        selected["P" + str(x)].append(df["Rating"][i])
    new_df = pd.DataFrame(selected)
    new_df["Average"] = new_df.mean(numeric_only=True, axis=1)
    new_df.to_csv("../data/Ratings.csv", index=False)



if __name__ == "__main__":
    select_ratings()