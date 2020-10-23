#!/usr/bin/env bash

## On many systems you will need to set up cuda:
#module load cuda
#conda activate tf-gpu
#export PYTHONUNBUFFERED=1

# the following script expects that the "DATA" directory contains training MSAs
# named "train10K", "train1M", and some of the steps expect there to also
# be precomputed files "target1000" (an MSA),  "bim_target100K.npy" (a 
# bivariate marginal file created using Mi3 based on 100K target sequences),
# and "top20_3000_target6M.db", created using HOM_r20.py (see below). "E_target1000.npy" is a numpy file containing the target energies of the target10000 sequences.

DATA=$PWD
MSAVAE_DIR=~/msavae
VAE=$MSAVAE_DIR/vaes.py

for latent in 2 4 6 8 10;
do

for size in 10K 1M
do

mkdir -p l${latent}_${size}
cd l${latent}_${size}

echo l$latent $size

# the following should be run on a gpu node
echo -e "\n\nTrain"
$VAE l${latent}_${size} train Church_VAE $DATA/train${size} $latent 250
echo -e "\n\nPlot latent"
$VAE l${latent}_${size} plot_latent $DATA/train${size}
echo -e "\n\nTVD"
$VAE l${latent}_${size} TVD $DATA/target10K
echo -e "\n\nEnergy"
$VAE l${latent}_${size} energy $DATA/target1000 --ref_energy $DATA/E_target1000.npy
echo -e "\n\nGen 6M"
$VAE l${latent}_${size} gen 6000000 -o gen_l${latent}_${size}_6M
head -n 100000 gen_l${latent}_${size}_6M >gen_l${latent}_${size}_100K
echo -e "\n\nCijab"
# the getMarginals script is provided with Mi3-GPU
# https://github.com/ahaldane/Mi3-GPU
getMarginals.py gen_l${latent}_${size}_100K bim_l${latent}_${size}
$MSAVAE_DIR/plotC.py $DATA/bim_target100K.npy bim_l${latent}_${size}.npy --name C_l${latent}_${size}.png

# this one should be run on cpu node
# the top20_3000_target6M.db file should be created separately using HOM_r20.py
# https://github.com/ahaldane/HOM_r20
echo -e "\n\nR20"
~/HOM_r20/HOM_r20.py count $DATA/top20_3000_target6M r20_l${latent}_${size} gen_l${latent}_${size}_6M

cd ..

done


done
