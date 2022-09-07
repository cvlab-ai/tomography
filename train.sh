#!/usr/bin/bash

nohup python ./src/run_training.py -o 8 50 1 0 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 2>normal_unet.log 1>normal_unet.err &
nohup python ./src/run_training.py -o 8 50 2 0 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 2>hard_tanh.log 1>hard_tanh.err &
nohup python ./src/run_training.py -o 8 50 3 0 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>adaptive_sigmoid.log 1>adaptive_sigmoid.err &
nohup python ./src/run_training.py -o 8 50 4 0 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_tanh" 2>adaptive_tanh.log 1>adaptive_tanh.err &