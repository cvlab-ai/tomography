#!/usr/bin/bash

nohup python ./src/run_training.py -o 8 50 1 0 0.00005 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 2>normal_unet00005.log 1>normal_unet00005.err &
nohup python ./src/run_training.py -o 8 50 2 0 0.00005 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 2>hard_tanh00005.log 1>hard_tanh00005.err &
nohup python ./src/run_training.py -o 8 50 3 0 0.00005 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>adaptive_sigmoid00005.log 1>adaptive_sigmoid00005.err &
nohup python ./src/run_training.py -o 8 50 4 0 0.00005 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_tanh" 2>adaptive_tanh00005.log 1>adaptive_tanh00005.err &
