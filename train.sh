#!/usr/bin/bash


nohup python ./src/run_training.py -o 8 50 1 0 0.00005 0.0 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>normal_unet00005_0.log 1>normal_unet00005_0.err &
nohup python ./src/run_training.py -o 8 50 2 0 0.00005 0.0 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 2>hard_tanh00005_0.log 1>hard_tanh00005_0.err &
nohup python ./src/run_training.py -o 8 50 3 0 0.00005 0.0 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_tanh" 2>normal_unet00005_0.log 1>normal_unet00005_0.err &
nohup python ./src/run_training.py -o 8 50 4 0 0.00005 0.0000001 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>normal_unet00005_0000001.log 1>normal_unet00005_0000001.err &
nohup python ./src/run_training.py -o 8 50 5 0 0.00005 0.0000001 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 2>hard_tanh00005_0000001.log 1>hard_tanh00005_0000001.err &
nohup python ./src/run_training.py -o 8 50 6 0 0.00005 0.0000001 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_tanh" 2>normal_unet00005_0000001.log 1>normal_unet00005_0000001.err &
