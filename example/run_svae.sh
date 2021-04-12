#!/usr/bin/env bash

## On many systems you will need to set up cuda:
#module load cuda
#conda activate tf-gpu
#export PYTHONUNBUFFERED=1

VAE=../vaes.py

latent=8
size=10K

echo -e "\n\nTrain"
$VAE l${latent}_${size} train sVAE train10K $latent 250
echo -e "\n\nPlot latent"
$VAE l${latent}_${size} plot_latent train10K
echo -e "\n\nTVD"
$VAE l${latent}_${size} TVD target10K
echo -e "\n\nGen 6M"
$VAE l${latent}_${size} gen 100000 -o gen_l${latent}_10K_100K
