#!/usr/bin/bash

nohup python ./src/run_training.py -o 8 50 1 0 0.001 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet001" 2>normal_unet001.log 1>normal_unet001.err &
nohup python ./src/run_training.py -o 8 50 2 0 0.005 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet005" 2>normal_unet005.log 1>normal_unet005.err &
nohup python ./src/run_training.py -o 8 50 3 0 0.0001 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet0001" 2>normal_unet0001.log 1>normal_unet0001.err &
nohup python ./src/run_training.py -o 8 50 4 0 0.005 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet0005" 2>normal_unet0005.log 1>normal_unet0005.err &
