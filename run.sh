#!/usr/bin/env bash
##PBS -q gpu
##PBS -l walltime=24:00:00,nodes=1
##PBS -N ds
##PBS -e stderr
##PBS -o stdout

#export PATH=$HOME/.local/bin:$PATH
#export PATH=$HOME/anaconda3/bin:$PATH
#export LD_LIBRARY_PATH=$HOME/.local/lib/../lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

#cd $PBS_O_WORKDIR

#module load cuda
#conda activate tf-gpu

#export PYTHONUNBUFFERED=1

DATA=~/VAE/vvae/vae_data
VAE=~/VAE/Church/fvae/vaes.py
model=Church_VAE

#for latent in 1 2 4 8 16;
for latent in 2
do

mkdir -p ${model}_l$latent
cd ${model}_l$latent

for size in 10K 1M
#for size in 10K
do

$VAE ${model}_l${latent}_k${size} train $model $DATA/train_${size} $latent 250
$VAE ${model}_l${latent}_k${size} plot_latent $DATA/train_${size}
$VAE ${model}_l${latent}_k${size} TVD $DATA/val_10K
$VAE ${model}_l${latent}_k${size} seq_accuracy ../../valseq3
$VAE ${model}_l${latent}_k${size} energy ../../valseq1000 --ref_energy ../../valE1000.npy
#$VAE ${model}_l${latent}_k${size} gen 6000000
#head -n 100000 gen_${model}_l${latent}_k${size}_6000000 >gen_${model}_l${latent}_k${size}_100000
$VAE ${model}_l${latent}_k${size} gen 100000
getMarginals.py gen_${model}_l${latent}_k${size}_100000 bim_l${latent}_k${size}
../plotC.py ../../bim_val1M.npy bim_l${latent}_k${size}.npy --name C_l${latent}_k${size}.png

done

cd ..

done
