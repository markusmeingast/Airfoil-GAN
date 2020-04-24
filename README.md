# Airfoil-GAN

# Data Scraping

The input data to train the above GAN architecture is derived from [Airfoil Tools](http://airfoiltools.com/). About 1600 airfoil shapes were scraped, and their respective crossection area calculated. Next to the shapes simulation results (CD and CL) were scraped for various angles of attacks. Specifically the case of Reynolds numbers at 200,000 were used.

The profiles were preprocessed to aid the training of the model. The following steps were implemented:

1. normalized to chord length of 1
1. used cosine spacing to refine the LE
1. Used the `Selig` profile formatting, interpolated to specified number of points
1. limited to airfoils with area smaller 0.2
1. supplying CL, CD and area as parameters

Scraping was done in parallel using `ray` on 20 CPUs.

# Data Exploration

The coefficient of lift shows a roughly bi-modal normal distribution, as seen below.  
![asd](imgs/cl_dist.png)

The coefficient of drag shows a Weibull-like distribution, limited by positive values only. For GAN performance, a close-to-normal distribution is beneficial. Log-scaling the coefficient produces a qualitatively similar distribution to the lift coefficient.
![asd](imgs/cd_dist.png)
![asd](imgs/cd_dist_log.png)

Mapping the two coefficient distributions against each other gives an interesting map. The bi-modal distributions are correlated in two distinct regions (Region $Q$ and $R$). $Q$ is localized to a relatively small area around $C_L=-0.5$ and $C_D=0.08$. Region $R$ is far more spread out, with $C_L$ between 0.0 and 1.5, and $C_D$ between 0.009 and 0.03.
![asd](imgs/cl_cd_dist.png)
Notably, very few samples exist within the center of this map with $C_L$ around 0.5 and $C_D$ about 0.05. This could be an excellent oportunity to check the GAN performance to explore this specificdesign space.

The area distribution follows also a roughly normal distribution and can be used straightforward as input to the GAN.
![asd](imgs/a_dist.png)

# Data Generator

The training process relies on a generator script to take profile samples and their respective parameters. Some processing is done at this stage to aid the training of the GAN model:

1. normalized the crossectional area to zero-mean and unit-standard deviation (scale factors: `a_mean = 0.085` and `a_std = 0.025`)
1. normalized CL to zero-mean and unit-standard deviation (scale factors: `cl_mean = 0.5` and `cl_std = 0.7`)
1. normalized log-scaled CD to zero-mean and unit-standard deviation (scale factors: `cl_mean = -3.6` and `cl_std = 0.7`)

# Model Architecture

Specific approaches/assumptions/effects were studied:

1. Using a Gaussian blur kernel as the last layer in `G` to smoothen the profile
1. Using a SG-kernel filter as the last layer in `G` to smoothen the profile
1. Using `mirrored` on `nearest` point padding in order to keep TE points close to the original prediction
1. Enforcing a TE closing condition
1. Batchnormalization did not show a benefit in the learning phase, even when only applied to generator. Using batch normalization for `G`, `D` and `E` showed highly unstable learning progress.
1. `LeakyReLU` activations seemed to show most stable learning characteristics, although improvements may be made reinvestigating `ReLU` or `ELU`
1. Label smoothing is used during `D`-training
1. `AE` is trained using `MAE`

# Training

Training was performed on a dual Xeon X5660 machine with a GTX 970 GPU. Total training time for 2000 epochs was roughly 18h.

# Validation

The validation is run by exporting a number of profiles generated for specific parameters, as shown below. The profiles are input into `XFoil` to follow the simulation methodology of the input data. The calculated `CL` and `CD` are compared against the user input.

To automate the validation process, initial steps have been taken to run `XFoil` in batch mode.
