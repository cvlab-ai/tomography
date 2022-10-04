#!/usr/bin/bash

nohup python ./src/run_training.py -o 16 50 1 0 0.00005 0.00005 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>mz_fix_test/normal_unet00005_00005.log 1>mz_fix_test/normal_unet00005_00005.err &
nohup python ./src/run_training.py -o 16 50 2 0 0.00005 0.00005 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 2>mz_fix_test/hard_tanh00005_00005.log 1>mz_fix_test/hard_tanh00005_00005.err &
nohup python ./src/run_training.py -o 16 50 3 0 0.00005 0.00005 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_tanh" 2>mz_fix_test/adaptive_tanh00005_00005.log 1>mz_fix_test/adaptive_tanh00005_00005.err &
nohup python ./src/run_training.py -o 16 50 4 0 0.00005 0.00005 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 2>mz_fix_test/normal_unet00005_00005.log 1>mz_fix_test/normal_unet00005_00005.err &