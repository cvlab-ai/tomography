#!/usr/bin/bash

python ./src/run_training.py 64 50 4 0 0.00005 0.00005 128 --tumor /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>logs/128_nobatch_tumor.log 1>logs/128_nobatch_tumor.err &
python ./src/run_training.py 64 50 4 0 0.00005 0.00005 256 --use_batch_norm  /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" 2>logs/256_batch_liver.log 1>logs/256_batch_liver.err &
