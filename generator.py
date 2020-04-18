"""
Generator to read in scraped profiles and return batches.
Returns:
 - X:   Curve data (BATCH_SIZE, POINTS, 2, 1) in cartesian coordinates
 - y:   Parameters (3, 1)
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import glob
import numpy as np

################################################################################
# %% DEFINE GENERATOR FUNCTION
################################################################################

def profile_generator(BATCH_SIZE=512, POINTS=64):

    ##### GET PROFILE FILES
    fimg = sorted(glob.glob(f'01-prep-data/{POINTS}/X_*.npy'))

    ##### READ FIRST FILE TO BUFFER
    #print(f'----> Reading {fimg[0]} <----')
    X = np.load(fimg[0])
    y = np.load(fimg[0].replace('X', 'y'))

    ##### SHUFFLE
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    ##### START COUNTING
    i = 0

    ##### LOOP INDEFINITELY
    while True:

        ##### IF BUFFER NOT SUFFICIENT TO COVER NEXT BATCH
        if len(X) < BATCH_SIZE:

            ##### SET NEXT INDEX BASED ON LENGHT OF FOLDER CONTENTS
            if i<len(fimg)-1:
                i += 1
            else:
                i = 0

            ##### APPEND NEXT FILE
            #print(f'----> Reading {fimg[i]} <----')
            X = np.concatenate((X, np.load(fimg[i])), axis=0)
            y = np.concatenate((y, np.load(fimg[i].replace('X', 'y'))), axis=0)

            ##### SHUFFLE
            idx = np.random.permutation(len(X))
            X = X[idx]
            y = y[idx]

        else:

            ##### SHIFT/SCALE TO -1.0 ... 1.0
            X[:BATCH_SIZE,: ,0, 0] = (X[:BATCH_SIZE,:, 0, 0] - 0.5)*2.0
            X[:BATCH_SIZE,: ,1, 0] = (X[:BATCH_SIZE,:, 1, 0])*2.0

            ##### NORMALIZE PARAMETERS (BASED ON DATASOURCE!)
            y[:BATCH_SIZE, 0] = y[:BATCH_SIZE, 0]/2.3
            y[:BATCH_SIZE, 1] = y[:BATCH_SIZE, 1]/0.28*2.0 - 1.0
            y[:BATCH_SIZE, 2] = y[:BATCH_SIZE, 2]/0.20*2.0 - 1.0

            ##### YIELD SET
            yield X[:BATCH_SIZE], y[:BATCH_SIZE]

            ##### REMOVE YIELDED RESULTS
            X = np.delete(X, range(BATCH_SIZE), axis=0)
            y = np.delete(y, range(BATCH_SIZE), axis=0)
