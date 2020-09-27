import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pdb

# load the data
digits = load_digits()

# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=15, whiten=False)
data = pca.fit_transform(digits.data)
orig_data = data[:44]


# save_data
#num_data = data.shape[0]
#labels = np.zeros(num_data).reshape(num_data, 1)
#savedata = np.concatenate([labels, data], axis=1)
#with open("train_data.txt", "wb") as f:
#  np.savetxt(f, savedata)
with open("race_train_l2lsh_sampled.txt" , "rb") as f:
  race_sampled_data = np.loadtxt(f)
race_sampled_data = race_sampled_data[:,1:]


# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

orig_score = kde.score(orig_data) / orig_data.shape[0]
# sample 44 new points from the data
new_data = kde.sample(44, random_state=0)
new_score = kde.score(new_data) / new_data.shape[0]
new_data = pca.inverse_transform(new_data)

# turn data into a 4x11 grid
new_data = new_data.reshape((4, 11, -1))
real_data = digits.data[:44].reshape((4, 11, -1))

race_score = kde.score(race_sampled_data) / race_sampled_data.shape[0]
race_sampled_data = pca.inverse_transform(race_sampled_data)
sampled_data = race_sampled_data[:44].reshape((4, 11, -1))

# plot real digits and resampled digits
fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    ax[4, j].set_visible(False)
    for i in range(4):
        im = ax[i, j].imshow(sampled_data[i, j].reshape((8, 8)),
                             cmap=plt.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)
        im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)

ax[0, 5].set_title('Race Sampled digits')
ax[5, 5].set_title('"New" digits drawn from the kernel density model')

print("Scores","orig:", orig_score, "new:", new_score, "race:", race_score)
plt.show()

