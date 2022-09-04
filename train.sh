#!/usr/bin/bash

nohup python ./src/run_training.py 8 100 0 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared > train0.log &
nohup python ./src/run_training.py 8 100 1 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared > train1.log &
nohup python ./src/run_training.py 8 100 2 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared > train2.log &
nohup python ./src/run_training.py 8 100 3 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared > train3.log &
nohup python ./src/run_training.py 8 100 4 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared > train4.log &