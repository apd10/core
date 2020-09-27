import numpy as np
import pickle
import sys
import os
import pdb

ignore_1 = False
fname = sys.argv[1]

if len(sys.argv) > 2:
  ignore_1 = int(sys.argv[2])

with open(fname, "rb") as f:
    d = pickle.load(f)

cs = d["params"]["num_classes"]
rs = d["params"]["repetitions"]
l = []
for c in range(cs):
    for r in range(rs):
        X = d["memory"][0][0]['topK']
        if ignore_1:
            X = X[X['value'] > 1]
        l.append(X.values.astype(int))
BS = np.concatenate(l)
#W = np.concatenate([np.array(d["hashfunction"]["W"]), np.array(d["hashfunction"]["b"]).reshape(1,-1) ])
#np.savez_compressed("w.npz", W)
np.savez_compressed("bs.npz", BS)
os.system("zip bs.zip bs.npz")
statinfo = os.stat("bs.zip")
print(statinfo.st_size // 10**5 / 10, "MB", statinfo.st_size // 10**2 / 10, "KB")

