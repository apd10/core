import pandas as pd
import sys
import numpy as np

if len(sys.argv) < 3:
    print("Usage: <script.py>  <fname> <sep> <skip_rows>")
    exit(0)
f = sys.argv[1]
sep = sys.argv[2]
skip_rows = int(sys.argv[3])


d = pd.read_csv(f, sep = sep, header=None, skiprows=skip_rows)
x = d.values[:,1:]
mu = np.mean(x, axis=0)
std = np.std(x, axis=0)

x = (x - mu)/std
norms = np.sqrt(np.sum(np.multiply(x, x), axis=1))
pctiles = np.array([50, 75, 90, 95, 99, 99.9])
norm_stats = np.array([np.percentile(norms, pct) for pct in pctiles])

writef = '/'.join(f.split('/')[:-1]) + '/centering_info.npz'
print(writef)
np.savez_compressed(writef, mu=mu, std=std, pct=pctiles,  post_norm_stats=norm_stats)
print(pctiles)
print(norm_stats)

