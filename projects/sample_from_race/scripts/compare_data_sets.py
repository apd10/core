'''
  This script is used to evaluate the samples vs the original data. 
'''
import argparse
import pandas as pd
import sklearn.datasets as ds
from sklearn.neighbors import KernelDensity
import pdb
import numpy as np
from sklearn.model_selection import GridSearchCV

args = argparse.ArgumentParser()
args.add_argument("--original", type=str, action="store", dest="original_file", required=True)
args.add_argument("--sampled", type=str, action="store", dest="sampled_file", required=True)
args.add_argument("--bandwidths", type=str, action="store", dest="bandwidth", default=None)
args.add_argument("--orig_type", type=str, action="store", dest="orig_type", default="csv")
args.add_argument("--sample_type", type=str, action="store", dest="sample_type", default="csv")
args.add_argument("--dim", type=int, action="store", dest="dim", required=True)
args.add_argument("--onlyNew", action="store_true", dest="onlyNew", default=False)


results = args.parse_args()
original_file = results.original_file
sampled_file = results.sampled_file
bandwidth = results.bandwidth
orig_type=results.orig_type
sample_type=results.sample_type
dim = results.dim
onlyNew = results.onlyNew

if orig_type == "csv":
    original_data = pd.read_csv(original_file, sep=' ')
    original_data.columns = ['label'] + [str(i) for i in np.arange(dim)]
elif orig_type == "svm":
    X,y = ds.load_svmlight_file(original_file, n_features=dim)
    original_data = pd.DataFrame(X.todense())
    original_data.columns = [str(i) for i in np.arange(dim)]
    original_data["label"] = y
    

if sample_type == "csv":
    sampled_data = pd.read_csv(sampled_file , sep=' ')
    sampled_data.columns = ['label'] + [str(i) for i in np.arange(dim)]
elif sample_type == "svm":
    X,y = ds.load_svmlight_file(sampled_file, n_features=dim)
    sampled_data = pd.DataFrame(X.todense())
    sampled_data.columns = [str(i) for i in np.arange(dim)]
    sampled_data["label"] = y


NUM_ORIG = len(original_data)
NUM_SAMP = len(sampled_data)

fcols = [c for c in original_data.columns if c !=  'label']
classes = np.sort(np.unique(sampled_data["label"].values))
kdes = {}
for c in classes:
  print("\n\n=========================","class", c,"============================\n\n")
  cdata = original_data.loc[original_data.label == c, fcols].values
  sampled_cdata = sampled_data.loc[sampled_data.label == c, fcols].values
  NUM_CDATA = cdata.shape[0]
  NUM_SDATA = sampled_cdata.shape[0]
  print("Fitting & Sampling...")
  if bandwidth is not None:
      for bw in [float(b) for b in bandwidth.split(",")]:
          print(" -- bw --", bw)
          kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(cdata)
          if not onlyNew:
              samples = kde.sample(NUM_SDATA, random_state=0)
    
              # random sampling.
              max_val = np.max(cdata)
              min_val = np.min(cdata)
              rnd_data = np.random.rand(NUM_SDATA, cdata.shape[1]) * (max_val - min_val) + min_val
              
              print("Evaluating ...")
              kdes[c] = kde
              odata = cdata[0:NUM_SDATA, :]
              print("class",c,"original", kde.score(odata) /  NUM_SDATA)
              print("class",c,"sklearn KDE sampled", kde.score(samples) /  NUM_SDATA)
              print("class",c,"random data", kde.score(rnd_data) / NUM_SDATA)
          print("class",c,"Provided sampled", kde.score(sampled_cdata) / NUM_SDATA)
  else:
      # use grid search cross-validation to optimize the bandwidth
      print("Figuring out bandwidth via cross validation")
      params = {'bandwidth': np.logspace(-2,0.5,10)}
      grid = GridSearchCV(KernelDensity(), params)
      grid.fit(cdata)
      print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
      # use the best estimator to compute the kernel density estimate
      kde = grid.best_estimator_

      samples = kde.sample(NUM_SDATA, random_state=0)
      print("Evaluating ...")
      kdes[c] = kde
      odata = cdata[0:NUM_SDATA, :]
      print("class", c,"original", kde.score(odata) /  NUM_SDATA)
      print("class",c,"sklearn KDE sampled", kde.score(samples) /  NUM_SDATA)
      print("class",c,"Provided sampled", kde.score(sampled_cdata) / NUM_SDATA)
