import pickle
import sys
import pdb

fname = sys.argv[1]
full=False
if len(sys.argv) > 2:
  full = sys.argv[2]

with open(fname, "rb") as f: 
    df = pickle.load(f) 

if len(sys.argv) > 3:
    pdb.set_trace()
class_counts = df["class_counts"]
pct = 0
den = 0
for c in range(len(class_counts)):
    for rep in range(df["params"]["repetitions"]):
        ct = df["memory"][c][rep]["topK"]["value"].sum()
        if full:
            print(c, rep, ct, "/", class_counts[c], "(",ct/class_counts[c]*100,")")
        pct += ct/class_counts[c]*100
        den += 1
print(pct/den)
