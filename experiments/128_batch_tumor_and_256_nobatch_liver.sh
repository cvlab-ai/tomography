#!/usr/bin/bash

python ./src/run_training.py 64 50 2 0 0.00005 0.00005 128 --use_batch_norm --tumor /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" "128_batch_tumor_hard_tanh" 2>logs/128_batch_tumor_hard_tanh.log 1>logs/128_batch_tumor_hard_tanh.err
python ./src/run_training.py 16 50 2 0 0.00005 0.00005 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" "256_nobatch_liver_hard_tanh" 2>logs/256_nobatch_liver_hard_tanh.log 1>logs/256_nobatch_liver_hard_tanh.err
python ./src/run_training.py 64 50 2 0 0.00005 0.00005 128 --use_batch_norm --tumor /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" "128_batch_tumor_normal_unet" 2>logs/128_batch_tumor_normal_unet.log 1>logs/128_batch_tumor_normal_unet.err
python ./src/run_training.py 16 50 2 0 0.00005 0.00005 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" "256_nobatch_liver_normal_unet" 2>logs/256_nobatch_liver_normal_unet.log 1>logs/256_nobatch_liver_normal_unet.err
