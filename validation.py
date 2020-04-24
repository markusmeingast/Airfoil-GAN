import numpy as np

import matplotlib.pyplot as mp

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.utils import CustomObjectScope

from SNConv2D import SpectralNormalization

from cgan import CGAN

from scipy.signal import savgol_filter

gan = CGAN()



g_model = load_model('02-results/g_model.h5', custom_objects={'edge_padding': gan.edge_padding, 'closing': gan.closing, 'kernel_init': gan.kernel_init, 'SpectralNormalization': SpectralNormalization})

# %%

##### TEST PARAMETERS

CL = 0.25
CD = 0.02
A = 0.07

LAT_DIM = 100

cl_mean = 0.50
cl_std =  0.7

cd_mean = -3.6
cd_std = 0.7

a_mean = 0.085
a_std = 0.025

nsamples = 5
cl = (CL-cl_mean)/cl_std*np.ones((nsamples))
cd = (np.log(CD)-cd_mean)/cd_std*np.ones((nsamples))
a = (A-a_mean)/a_std*np.ones((nsamples))
y_pred = np.array([
        cl,
        cd,
        a
    ]).T
z_pred = np.random.randn(nsamples, LAT_DIM)
X_pred = g_model.predict([y_pred, z_pred])

X_smooth = X_pred.copy()
X_smooth = savgol_filter(X_smooth, 9, 2, mode='nearest', axis=1)

for i in range(1):
    mp.plot(X_pred[i,:,0,0]+i*2.1,X_pred[i,:,1,0])
    mp.plot(X_smooth[i,:,0,0]+i*2.1, X_smooth[i,:,1,0], 'o-')
mp.axis('equal')
mp.title(f'CL: {cl[0]*0.7+0.5}, CD: {np.exp(cd[0]*0.7-3.6)}')
#mp.savefig(f'02-results/gen_{epoch:03d}.png')
#mp.close()

data = X_smooth[0, :, :, 0].copy()

data[:, 0] = data[:, 0]/2+0.5
data[:, 1] = data[:, 1]/2


np.savetxt('profile.dat', data)


# %%

import numpy as np
import matplotlib.pyplot as mp
import seaborn as sn

X = np.load('01-prep-data/32/X_000.npy')
data = np.load('01-prep-data/32/y_000.npy')

for i in range(1, 20):
    X = np.concatenate((X, np.load(f'01-prep-data/32/X_{i:03d}.npy')), axis=0)
    data = np.concatenate((data, np.load(f'01-prep-data/32/y_{i:03d}.npy')), axis=0)

#sn.distplot(data[:,2], bins=20)
sn.jointplot(data[:,0], np.log(data[:,1]), kind="hex", color="#4CB391")
#mp.title('$C_L$ vs. $C_D$ distribution')
mp.xlabel('$C_L$')
mp.ylabel('$C_D$')

mp.savefig('cl_cd_dist.png')


#sns.set(style="ticks")


idxcl = np.argwhere(np.logical_and(data[:,0]>0.0, data[:,0]<1.5))
idxcd = np.argwhere(np.logical_and(data[:,1]>0.009, data[:,1]<0.03))
idx = np.intersect1d(idxcl, idxcd)

for i in idx:
    mp.plot(X[i, :, 0, 0], X[i, :, 1, 0],'gray')
mp.axis('equal')
