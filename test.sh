#!/usr/bin/bash

nohup python ./src/run_testing.py -o 8 1 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" unet-hard-tanh-window-fold-0-loss-5e-05 2>normal_unet00005.log 1>normal_unet00005.err &
nohup python ./src/run_testing.py -o 8 2 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" unet-hard-tanh-window-fold-0-loss-5e-05 2>hard_tanh00005.log 1>hard_tanh00005.err &
nohup python ./src/run_testing.py -o 8 3 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" unet-adaptive-sigmoid-window-fold-0-loss-5e-05 2>adaptive_sigmoid00005.log 1>adaptive_sigmoid00005.err &
nohup python ./src/run_testing.py -o 8 4 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_tanh" unet-adaptive-tanh-window-fold-0-loss-5e-05 2>adaptive_tanh00005.log 1>adaptive_tanh00005.err &
