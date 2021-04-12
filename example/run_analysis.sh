#!/usr/bin/env bash

latent=8
size=10K

# the getMarginals script is provided with Mi3-GPU
# https://github.com/ahaldane/Mi3-GPU
getMarginals.py gen_l${latent}_10K_100K bim_l${latent}_10K
getMarginals.py target10K bim_target_10K
../plotC.py bim_target_10K.npy bim_l${latent}_10K.npy --name C_l${latent}_10K.png
