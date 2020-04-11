"""
This script generate NACA 4-digit (p,d,tt) airfoils by means of spline interpolation.
Spline control points, as well as lift and drag coefficients are passed in batches.
Using zero-thickness TE for now.

Nomenclature:
    - m: max camber (0 - 9.5% of chord)
    - p: camber position (0 - 90% of chord)
    - t: thickness (1 - 40% of chord)
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import matplotlib.pyplot as mp
from sim import coef
from tqdm import tqdm
import ray

num_cpus = 2
ray.init(num_cpus=num_cpus)

################################################################################
# %% FUNCTION
################################################################################

def camber_line(p, m, points=100):

    ##### INIT LOCAL POSITION VECTORS

    loc = (1.0-np.cos(np.linspace(0, np.pi, points)))/2
    yc = np.zeros((points,), dtype=float)

    ##### CALCULATE CENTER LINE
    for i in range(points):
        x = loc[i]
        if x<p:
            yc[i] = (m/p**2)*(2*p*x-x**2)
        else:
            yc[i] = (m/(1-p)**2)*((1-2*p)+2*p*x-x**2)

    return yc

def thickness(t, points=100):

    ##### INIT LOCAL POSITION VECTORS
    loc = (1.0-np.cos(np.linspace(0, np.pi, points)))/2
    yt = np.zeros((points,), dtype=float)

    ##### CALCULATE THICKNESS
    for i in range(points):
        x = loc[i]
        yt[i] = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    return yt

def naca_profile(m, p, t, points=100, angle=0):

    ##### SCALE PARAMETERS
    p = p/10
    m = m/100
    t = t/100

    ##### INIT LOCAL POSITION VECTOR
    loc = (1.0-np.cos(np.linspace(0, np.pi, points)))/2
    xu = np.zeros((points,), dtype=float)
    xl = np.zeros((points,), dtype=float)
    yu = np.zeros((points,), dtype=float)
    yl = np.zeros((points,), dtype=float)

    ##### GET CAMBER LINE AND THICKNESS DISTRIBUTION
    yc = camber_line(p, m, points)
    yt = thickness(t, points)

    ##### CALCULATE NORMAL ANGLE
    theta = np.zeros((points,), dtype=float)
    for i in range(points):
        x = loc[i]

        if x<p:
            theta[i] = np.arctan((2*m/p**2)*(p-x))
        else:
            theta[i] = np.arctan((2*m/(1-p)**2)*(p-x))

    ##### PROFILE CALCULATION
    for i in range(points):
        x = loc[i]
        xu[i] = x - yt[i]*np.sin(theta[i])
        xl[i] = x + yt[i]*np.sin(theta[i])
        yu[i] = yc[i] + yt[i]*np.cos(theta[i])
        yl[i] = yc[i] - yt[i]*np.cos(theta[i])

    ##### SET ANGLE OF ATTACK (ASSUMING CENTER AT 0.4)
    xu += -0.4
    xl += -0.4
    angle = angle*np.pi/180.0

    xnewu = xu.copy()
    xnewl = xl.copy()
    ynewu = yu.copy()
    ynewl = yl.copy()

    for i in range(points):
        xnewu[i] =  xu[i]*np.cos(angle) - yu[i]*np.sin(angle)
        xnewl[i] =  xl[i]*np.cos(angle) - yl[i]*np.sin(angle)
        ynewu[i] =  xu[i]*np.sin(angle) + yu[i]*np.cos(angle)
        ynewl[i] =  xl[i]*np.sin(angle) + yl[i]*np.cos(angle)

    xu = xnewu
    xl = xnewl
    yu = ynewu
    yl = ynewl

    xm = (xu.mean()+xl.mean())/2
    ym = (yu.mean()+yl.mean())/2

    xu = xu-xm
    xl = xl-xm
    yu = yu-ym
    yl = yl-ym

    ##### SORT AND COMBINE
    return np.concatenate((np.flip(xl), xu)), np.concatenate((np.flip(yl), yu))

# %%

samples = 200
points = 100
DEBUG = False

@ray.remote
def par_func(samples, points, DEBUG):

    X = np.zeros((samples, points//2, 2, 1), dtype=float)
    Y = np.zeros((samples, 2), dtype=float)
    N = np.zeros((samples, 4), dtype=float)

    for sample in range(samples):

        ##### BUILD RANDOM AIRFOILS
        ##### MAX CAMBER
        ml = 0.0
        mu = 5.0
        m = (mu-ml)*np.random.random()+ml

        ##### MAX CAMBER LOC
        pl = 0.5
        pu = 5.0
        p = (pu-pl)*np.random.random()+pl

        ##### THICKNESS
        tl = 1.0
        tu = 40.0
        t = (tu-tl)*np.random.random()+tl

        ##### ANGLE
        al = -20.0
        au = 20.0
        a = (au-al)*np.random.random()+al

        if DEBUG:
            print(m, p, t, a)

        ##### BUILD PROFILE
        x, y = naca_profile(m, p, t, 200, a)
        if DEBUG:
            mp.plot(x,y)
            mp.axis([0,1,0,1])
            mp.axis('equal')
            mp.show()

        ##### CALCULATE CL AND CD
        cd, cl = coef(x, y, DEBUG)
        if DEBUG:
            print(cl, cd)

        #####
        X[sample, :, 0, 0] = x[::8]
        X[sample, :, 1, 0] = y[::8]

        Y[sample, 0] = cl
        Y[sample, 1] = cd

        N[sample, :] = [m, p, t, a]

    return X, Y, N

object_id = [par_func.remote(samples//num_cpus, points, DEBUG) for i in range(num_cpus)]
out = ray.get(object_id)

X = out[0][0]
Y = out[0][1]
N = out[0][2]

for i in range(1, num_cpus):
    X = np.concatenate((X, out[i][0]), axis=0)
    Y = np.concatenate((Y, out[i][1]), axis=0)
    N = np.concatenate((N, out[i][2]), axis=0)

np.save('X_000.npy', X)
np.save('y_000.npy', Y)
np.save('n_000.npy', N)
