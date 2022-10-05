#!/usr/bin/bash

python ./src/run_training.py 64 50 3 0 0.00005 0.00005 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>logs/128_nobatch_liver.log 1>logs/128_nobatch_liver.err &
python ./src/run_training.py 64 50 3 0 0.00005 0.00005 256 --use_batch_norm --tumor /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>logs/256_batch_tumor.log 1>logs/256_batch_tumor.err &
