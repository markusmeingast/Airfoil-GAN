"""
Assumptions:
 - X:   curve data (range -1.0 ... 1.0)
 - y:   parameters (-1.0 ... 1.0)
 - z:   noise (normal centered around 0)
 - w:   truth (-1.0/fake - 1.0/real)
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
from generator import profile_generator
from cgan import CGAN
#from acgan import ACGAN
import matplotlib.pyplot as mp
import tensorflow as tf

################################################################################
# %% CONSTANTS
################################################################################

EPOCHS = 200
BATCH_SIZE = 1024
BATCHES = 160
POINTS = 64
DAT_SHP = (POINTS, 2, 1)
LAT_DIM = 100
PAR_DIM = 3
DEPTH = 32
LEARN_RATE = 0.0002
RESTART = False

################################################################################
# %% KERAS/TF SETTINGS
################################################################################

##### ALLOW GPU MEMORY GROWTH
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

##### USE MIXED PRECISION WHERE POSSIBLE
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

################################################################################
# %% START GENERATOR
################################################################################

gen = profile_generator(BATCH_SIZE, POINTS)

################################################################################
# %% BUILD GAN MODEL
################################################################################

gan = CGAN(DAT_SHP=DAT_SHP, PAR_DIM=PAR_DIM, LAT_DIM=LAT_DIM, DEPTH=DEPTH, LEARN_RATE=LEARN_RATE)

g_model = gan.build_generator()
d_model = gan.build_discriminator()

if RESTART:
    g_model.load_weights('02-results/g_model.h5')
    d_model.load_weights('02-results/d_model.h5')

print(g_model.summary())
print(d_model.summary())

gan_model = gan.build_gan(g_model, d_model)

################################################################################
# %% LOOP EPOCHS
################################################################################

if RESTART:
    loss = np.load('02-results/loss.npy').tolist()
    acc = np.load('02-results/acc.npy').tolist()
else:
    acc = []
    loss = []

##### LOOP OVER EPOCHS
for epoch in range(EPOCHS):

    ##### LOOP OVER BATCHES
    for batch in range(BATCHES):

        ##### GET REAL DATA
        X_real, y_real = next(gen)
        w_real = np.ones((len(y_real),1), dtype=float)

        ##### PRODUCE FAKE DATA
        w_fake = np.zeros((len(y_real),1), dtype=float)
        y_fake = np.random.uniform(-1, 1, (BATCH_SIZE, PAR_DIM))
        z_fake = np.random.randn(BATCH_SIZE, LAT_DIM)
        X_fake = g_model.predict([y_fake, z_fake])

        ##### TRAIN DISCRIMINATOR
        d_loss_real = d_model.train_on_batch([X_real[:BATCH_SIZE//2], y_real[:BATCH_SIZE//2]], w_real[:BATCH_SIZE//2])
        d_loss_fake = d_model.train_on_batch([X_fake[:BATCH_SIZE//2], y_fake[:BATCH_SIZE//2]], w_fake[:BATCH_SIZE//2])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        ##### TRAIN GENERATOR
        g_loss = gan_model.train_on_batch([y_fake, z_fake], w_real)

        acc.append([d_loss[-1], g_loss[-1]])
        loss.append([d_loss[0], g_loss[0]])

    ##### PRINT PROGRESS
    print(f'Epoch: {epoch} - D loss: {d_loss[0]} - G loss: {g_loss[0]} - D(w) acc: {d_loss[-1]} - G(w) acc: {g_loss[-1]}')

    ############################################################################
    # %% TEST GENERATOR
    ############################################################################

    nsamples = 5
    cl = np.random.uniform(-1, 1, (1))*np.ones((nsamples))
    cd = np.random.uniform(-1, 1, (1))*np.ones((nsamples))
    y_pred = np.array([
        cl,
        cd,
        np.linspace(-1, 1, nsamples)
    ]).T
    z_pred = np.random.randn(nsamples, LAT_DIM)
    X_pred = g_model.predict([y_pred, z_pred])
    for i in range(5):
        mp.plot(X_pred[i,:,0,0]+i*2.1,X_pred[i,:,1,0]-0.5)
        mp.plot(X_real[i,:,0,0]+i*2.1,X_real[i,:,1,0]+0.5)
    mp.axis('equal')
    mp.title(f'CL/CD: {cl[0]*2.3} / {(cd[0]+1.0)*0.28/2.0}')
    mp.savefig(f'02-results/gen_{epoch:03d}.png')
    mp.close()

    ############################################################################
    # %% SAVE MODELS
    ############################################################################

    g_model.save('02-results/g_model.h5')
    d_model.save('02-results/d_model.h5')

    ############################################################################
    # %% PLOT LOSS CURVES
    ############################################################################

    fig = mp.figure(figsize=(10,8))
    mp.semilogy(np.array(loss))
    mp.xlabel('batch')
    mp.ylabel('loss')
    mp.legend(['D(w) loss', 'D(G(w)) loss'])
    mp.savefig('02-results/loss.png')
    mp.close()
    np.save('02-results/loss.npy', np.array(loss))

    ############################################################################
    # %% PLOT ACCURACY CURVES
    ############################################################################

    fig = mp.figure(figsize=(10,8))
    mp.plot(np.array(acc))
    mp.xlabel('batch')
    mp.ylabel('accuracy')
    mp.legend(['D(w) acc', 'D(G(w)) acc'])
    mp.savefig('02-results/acc.png')
    mp.close()
    np.save('02-results/acc.npy', np.array(acc))
