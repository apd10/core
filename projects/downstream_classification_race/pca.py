import sklearn.datasets as ds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import sys
import pickle

if len(sys.argv) < 2:
    print("usage: script.py train.txt orig_dim new_dim")
    exit(0)

orig_train = sys.argv[1]
orig_dim = int(sys.argv[2])
new_dim = int(sys.argv[3])
pca_train = orig_train.replace("train.txt", "pca_train_"+str(new_dim)+".txt")
pca_pickle = orig_train.replace("train.txt", "pca_"+str(new_dim)+".pickle")

with open(orig_train, "rb") as f:
    od = ds.load_svmlight_file(f)
d = od[0]
d = d.todense()
if d.shape[1] < orig_dim:
    z = np.zeros((d.shape[0], orig_dim - d.shape[1]))
    d = np.concatenate([d, z], axis=1)

pca = PCA(n_components=new_dim, whiten=False)
data = pca.fit_transform(d)
with open(pca_pickle, "wb") as f:
    pickle.dump(pca, f)

labels = od[1].reshape(-1,1)
np.concatenate([labels, data], axis=1)
ndata = np.concatenate([labels, data], axis=1)
np.savetxt(pca_train, ndata)
