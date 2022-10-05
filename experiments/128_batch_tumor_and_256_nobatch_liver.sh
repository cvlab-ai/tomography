#!/usr/bin/bash

python ./src/run_training.py 64 50 2 0 0.00005 0.00005 128 --use_batch_norm --tumor /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" "128_batch_tumor" 2>logs/128_batch_tumor.log 1>logs/128_batch_tumor.err
python ./src/run_training.py 16 50 2 0 0.00005 0.00005 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "adaptive_sigmoid" "256_nobatch_liver" 2>logs/256_nobatch_liver.log 1>logs/256_nobatch_liver.err
