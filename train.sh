#!/usr/bin/bash
source /home/macierz/s175573/tomography/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175573/tomography/tomography

nohup ./experiments/128_batch_liver_and_256_nobatch_tumor.sh &
nohup ./experiments/128_batch_tumor_and_256_nobatch_liver.sh &
nohup ./experiments/128_nobatch_liver_and_256_batch_tumor.sh &
nohup ./experiments/128_nobatch_tumor_and_256_batch_liver.sh &
