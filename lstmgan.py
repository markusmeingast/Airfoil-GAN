"""
This is an implementation of a conditional GAN (CGAN) network. The generator
predicts and image (in this case the number of points along the profile in x and
y).This is done by combining parameter terms as well as random noise, as well as
employing a gaussian blur to smooth the profile.

The discriminator receives the image as well as the parameter vector. These are
combined and finally tested for validity.

Losses are based on binary crossentropy only, i.e. the validity of results.

Issues were encountered using batch normalization. For now, it is ignored.


Observations:
 - Concatenating target parameters with image prior to Conv2D layers produces
   unstable solution. Parameter dependency seems present, but no clean solution obtained.
 - Concatenating target parameters with processed image data, post Conv2D, results
   in clean shapes, but no parameter dependency obtained (tested 1000+ epochs)
 - LeakyReLU after late concatenation produces weird results
 - BN for all Conv2D layers produces weird results.
 - BN on D only produces weird results
 - SN on all Conv2D layers seems to work fine, no parameter dependency as before
 - Reduced learning rate seems to stabilize slightly, but issue still not avoided



Guidelines:
 - smoothing works
 - no BN on D
 - BN on G
 - SN on all Conv2D layers (not restartable...)
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Activation, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Dense, BatchNormalization, Conv2D
from tensorflow.keras.layers import GaussianNoise, Dropout, LeakyReLU, Flatten, ReLU
from tensorflow.keras.layers import Lambda, LSTM, Bidirectional
from SNConv2D import SpectralNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K

################################################################################
# %% DEFINE INFOGAN CLASS
################################################################################

class LSTMGAN():

    """
    Implementation of the CGAN network with curve smoothing in the generator
    """

    ##### CLASS INIT
    def __init__(self, DAT_SHP=(64, 2, 1), PAR_DIM=3, LAT_DIM=100, DEPTH=32, LEARN_RATE=0.0002):

        """
        Initializing basic settings
        """

        ##### TARGET DATA SOURCE
        self.DAT_SHP = DAT_SHP
        self.PAR_DIM = PAR_DIM
        self.LAT_DIM = LAT_DIM
        self.DEPTH = DEPTH
        self.init = RandomNormal(mean=0.0, stddev=0.02)
        self.LEARN_RATE = LEARN_RATE
        self.optimizer = Adam(lr=self.LEARN_RATE, beta_1=0.5)
        #self.optimizer = RMSprop(lr=self.LEARN_RATE, clipnorm=1.)
        #self.optimizer = SGD(learning_rate=self.LEARN_RATE, momentum=0.1)
        self.BLUR = True
        self.CLOSE = True

    ##### GAUSSIAN BLUR FILTER (ISSUES AT END POINTS)
    def kernel_init(self, shape, dtype=float, partition_info=None):

        """
        Definition of a length 7 gaussian blur kernel to be used to smooth the profile
        """

        kernel = np.zeros(shape=shape)
        kernel[:,:,0,0] = np.array([[0.006],[0.061],[0.242],[0.383],[0.242],[0.061],[0.006]])
        return kernel

    ##### PAD EDGES TO MAKE GAUSSIAB BLUR WORK
    def edge_padding(self, X):

        """
        Custom padding layer to be called by Lambda. Adds each end point 3 times
        respectively to produce cleaner edge conditions
        """

        ##### PAD START
        Xlow0 = X[:, 0, :, :]
        Xlow1 = (2.0*Xlow0 - X[:, 1, :, :])[:, np.newaxis, :, :]
        Xlow2 = (2.0*Xlow0 - X[:, 2, :, :])[:, np.newaxis, :, :]
        Xlow3 = (2.0*Xlow0 - X[:, 3, :, :])[:, np.newaxis, :, :]

        ##### PAD END
        Xhigh0 = X[:, -1, :, :]
        Xhigh1 = (2.0*Xhigh0 - X[:, -2, :, :])[:, np.newaxis, :, :]
        Xhigh2 = (2.0*Xhigh0 - X[:, -3, :, :])[:, np.newaxis, :, :]
        Xhigh3 = (2.0*Xhigh0 - X[:, -4, :, :])[:, np.newaxis, :, :]

        ##### BUILD AND RETURN PADDED ARRAY
        X = K.concatenate((Xlow3,Xlow2,Xlow1,X,Xhigh1,Xhigh2,Xhigh3), axis=1)
        return X

    def closing(self, X):

        Xlow = X[:, 0, :, :][:, np.newaxis, :, :]
        Xhigh = X[:, -1, :, :][:, np.newaxis, :, :]
        Xmean = (Xlow+Xhigh)*0.5
        return K.concatenate((Xmean, X[:, 1:-1, :, :], Xmean), axis=1)

    def build_generator(self):

        """
        Generator network:
         - Input dimensions: (PAR_DIM)+(LAT_DIM)
         - Output dimensions: (DAT_SHP)
        """

        ##### INPUT LAYERS
        y_in = Input(shape=self.PAR_DIM)
        z_in = Input(shape=self.LAT_DIM)

        ##### COMBINE AND DENSE
        net = concatenate([y_in, z_in], axis=-1)
        net = Reshape((self.DAT_SHP[0], 1))(net)

        net = Bidirectional(LSTM(16, return_sequences=True))(net)
        net = Bidirectional(LSTM(8, return_sequences=True))(net)
        net = Bidirectional(LSTM(4, return_sequences=True))(net)
        net = Bidirectional(LSTM(2, return_sequences=True))(net)
        net = Bidirectional(LSTM(1, return_sequences=True))(net)

        X_out = Reshape(self.DAT_SHP)(net)

        ##### BUILD MODEL
        model = Model(inputs=[y_in, z_in], outputs=X_out)
        return model

    def build_discriminator(self):

        """
        Input dimensions: (DAT_SHP)
        Output dimensions: (1) + (PAR_DIM)
        """

        ##### INPUT LAYERS
        X_in = Input(self.DAT_SHP)
        y_in = Input(self.PAR_DIM)

        ##### ADD NOISE TO IMAGE
        Xnet = GaussianNoise(0.05)(X_in)

        ynet = Dense(self.DAT_SHP[0])(y_in)
        ynet = Reshape((self.DAT_SHP[0], 1, 1))(ynet)
        net = concatenate([Xnet, ynet], axis=2)
        net = Reshape((self.DAT_SHP[0], 3))(net)

        net = Bidirectional(LSTM(2, return_sequences=True))(net)
        net = Bidirectional(LSTM(4, return_sequences=True))(net)
        net = Bidirectional(LSTM(8, return_sequences=True))(net)
        net = Bidirectional(LSTM(16, return_sequences=True))(net)

        net = Flatten()(net)
        #net = Dense(64)(net)

        ##### VALIDITY
        w_out = Dense(1, activation='sigmoid')(net)

        ##### MODEL
        model = Model(inputs=[X_in, y_in], outputs=w_out)
        model.compile(loss=[BinaryCrossentropy(label_smoothing=0.5)], metrics=['accuracy'], optimizer=Adam(lr=self.LEARN_RATE, beta_1=0.5))

        return model

    def build_gan(self, g_model, d_model):

        """
        GAN network combined generator and discriminator networks for generator
        training. The discriminator is not trained within this model
        """

        ##### TRAIN ONLY GENERATOR
        d_model.trainable = False

        ##### INPUT LAYERS
        y_in = Input(shape=self.PAR_DIM)
        z_in = Input(shape=self.LAT_DIM)

        ##### GENERATE IMAGE
        X = g_model([y_in, z_in])

        ##### TEST IMAGE
        w_out = d_model([X, y_in])

        ##### BUILD, COMPILE AND RETURN MODEL
        gan_model = Model(inputs = [y_in, z_in], outputs = [w_out])
        gan_model.compile(loss=['binary_crossentropy'], metrics=['accuracy'], optimizer=Adam(lr=self.LEARN_RATE, beta_1=0.5))

        return gan_model
