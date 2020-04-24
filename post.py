################################################################################
# %% PLOTTING STUFF
################################################################################

import numpy as np
import matplotlib.pyplot as mp
import seaborn as sn

X = np.load('01-prep-data/32/X_000.npy')
data = np.load('01-prep-data/32/y_000.npy')

for i in range(1, 20):
    X = np.concatenate((X, np.load(f'01-prep-data/32/X_{i:03d}.npy')), axis=0)
    data = np.concatenate((data, np.load(f'01-prep-data/32/y_{i:03d}.npy')), axis=0)

#sn.distplot(data[:,2], bins=20)
g = sn.jointplot(data[:,0], np.log(data[:,1]), kind="hex", color="#4CB391")
ax = g.ax_joint
ylabels = ['0.01', '0.02', '0.03', '0.05', '0.07', '0.10', '0.15', '0.20']
yticks = [np.log(float(x)) for x in ylabels]
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
mp.xlabel('$C_L$')
mp.ylabel('$C_D$')

mp.savefig('cl_cd_dist.png')


#sns.set(style="ticks")

# %%

idxcl = np.argwhere(np.logical_and(data[:,0]>0.0, data[:,0]<1.5))
idxcd = np.argwhere(np.logical_and(data[:,1]>0.009, data[:,1]<0.03))
idx = np.intersect1d(idxcl, idxcd)

for i in idx:
    mp.plot(X[i, :, 0, 0], X[i, :, 1, 0],'gray')
mp.axis('equal')
