Description
===========

(Hopefully) faster HOM analysis by better parallelization and using a better data structure for histogram counting.

Instructions
============

Compile by running "make". Requires "seqload" module from Mi3 to be in python-path, and will run best on unix systems (osx/linux).

Currently set up to process 3 sequence datasets: data with weights, model, and indep. You may need to modify for other purposes.

Run as:

$ nreps=128
$ Nsample=16049
$ ./HOM_r20.py $nreps $Nsample data_seq model_seq indep_seq data_weights.npy

nreps is how many random x-mers to sample for each length L.
Nsample is how many of the model sequences to use as a small "finite sample" dataset to measure finite-sampling error.

The weights at the last command-line argument are optional, if you leave it out no weights are used.

The script is currently hardcoded to compute marginals for sets of positions (L) in range(2,10), you can change this my modifying the npos_range at the bottom of the .py file.

The script will output to the screen the progress it has made, and will output the results to a set of files r20_* where * is the L for that file. These files contain lines with 3 numbers, which are: 1. r20 data vs model, 2. r20 data vs indep, and 3. r20 finite-sample vs model.
