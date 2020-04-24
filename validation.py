"""
Script to run various predictions through the trained model and validate against
XFoil simulations.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import matplotlib.pyplot as mp
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.utils import CustomObjectScope
from SNConv2D import SpectralNormalization
from cgan import CGAN
from scipy.signal import savgol_filter
import os
import argparse

################################################################################
# %% TEST PARAMETERS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("CL", help="target lift coefficient (float ~-1.5 -- 2)")
parser.add_argument("CD", help="target drag coefficient (float ~0.01 -- 0.2)")
parser.add_argument("A", help="target crossectional area (float ~0.02 -- 0.2)")
args = parser.parse_args()

CL = float(args.CL)
CD = float(args.CD)
A = float(args.A)
LAT_DIM = 100

################################################################################
# %% INIT CGAN MODEL TO PASS CUSTOM LAYERS TO GENERATOR MODEL
################################################################################

gan = CGAN()
g_model = load_model('02-results/g_model.h5', custom_objects={'edge_padding': gan.edge_padding, 'closing': gan.closing, 'kernel_init': gan.kernel_init, 'SpectralNormalization': SpectralNormalization})

################################################################################
# %% SCALE FACTORS
################################################################################

cl_mean = 0.50
cl_std =  0.7

cd_mean = -3.6
cd_std = 0.7

a_mean = 0.085
a_std = 0.025

################################################################################
# %% RUN THROUGH GENERATOR
################################################################################

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
X_smooth = savgol_filter(X_smooth, 3, 2, mode='nearest', axis=1)

"""
for i in range(1):
    mp.plot(X_pred[i,:,0,0]+i*2.1,X_pred[i,:,1,0])
    mp.plot(X_smooth[i,:,0,0]+i*2.1, X_smooth[i,:,1,0], 'o-')
mp.axis('equal')
mp.title(f'CL: {cl[0]*0.7+0.5}, CD: {np.exp(cd[0]*0.7-3.6)}')
#mp.savefig(f'02-results/gen_{epoch:03d}.png')
#mp.close()
"""

################################################################################
# %% EXPORT PROFILE FOR XFOIL
################################################################################

data = X_smooth[0, :, :, 0].copy()
data[:, 0] = data[:, 0]/2+0.5
data[:, 1] = data[:, 1]/2
np.savetxt('profile.dat', data)

os.system('xfoil < script.xf')
os.system('evince plot.ps')
