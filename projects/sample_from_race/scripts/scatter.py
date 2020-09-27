import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
fname = sys.argv[1]

data = pd.read_csv(fname, header="infer", sep=" ")
data.columns = ["label", "x", "y"]
labels = data["label"].astype(int).unique()
colors = ["r", "b", "g", "y"]

print (labels)
for l in labels:
  d = data.loc[(data["label"] == l), ["x", "y"]]
  plt.scatter(d["x"].values, d["y"].values, color=colors[l])
#plt.savefig(fname.replace(".txt", ".png"))
plt.show()
