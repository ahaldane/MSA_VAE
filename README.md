Description
===========

Code to run and validate VAEs.

Builds on the code in https://github.com/samsinai/VAE_protein_function which is under MIT license.


Setup
=====

Run make to compule the helper module. After compiling, a file seqtools.xxx.so should have been created.


Usage
=====

The vaes.py script is the main tool, and will output usage info when run.

Run the vaes.py script in one of the following ways:

To train a new model:
---------------------

    $ vaes.py my_name train Church_VAE trainingMSA 2 250

"my_name" is a name for use as prefix for output files (pickled VAE). 
"train" tells the script to run in training mode. 
"Church_VAE" tells the script to use the Church VAE (can also be VVAE, Deep_VAE)
"trainingMSA" is the file (MSA) to train on
"2" is the number of latent dimensions
"250" is the size of the encoder/decoder layers for the church VAE (only necessary for Church_VAE, other vaes have different options here)

Optional arg:
 "--TVDseqs msafile" will track the TVD of the hamming distance distribution on each epock, and make a plot.

 Will create plots of the loss function over epochs, and a csv file containing the loss over epichs.


To generate sequences:
----------------------

    $ vaes.py my_name gen 100000

"my_name" is the name prefix for pickled VAE & output files.
"gen" tells the script to run in sequence generation mode. 
1000000 is the # of sequences to generate

To compute energies
-------------------

    $ vaes.py my_name energy MSAfile --ref_energy E.npy

"my_name" is the name prefix for pickled VAE & output files.
"energy" tells the script to run in energy computation mode. 
"MSAfile" is the MSA to compute energies for
--ref_energy (optional) is reference energies to include in comparison plot


To Plot the Latent Space
------------------------

    $VAE my_name plot_latent trainingMSA

Will create some png files of the latent space.

Other computations:
------------------------

See the "run.sh" file. Can also compute TVD plot, C correlation plot, and more.

