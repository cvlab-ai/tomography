#!/usr/bin/bash
source /home/macierz/s175573/tomography/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175573/tomography/tomography

nohup $(python ./src/run_testing.py 64 1 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 128_batch_liver_hard_tanh --use_batch_norm 2>test_logs/128_batch_liver_hard_tanh.log 1>test_logs/128_batch_liver_hard_tanh.err; \
        python ./src/run_testing.py 16 1 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 256_batch_liver_hard_tanh --use_batch_norm 2>test_logs/256_batch_liver_hard_tanh.log 1>test_logs/256_batch_liver_hard_tanh.err; \
        python ./src/run_testing.py 64 1 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 128_batch_liver_normal_unet --use_batch_norm 2>test_logs/128_batch_liver_normal_unet.log 1>test_logs/128_batch_liver_normal_unet.err;  \
        python ./src/run_testing.py 16 1 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 256_batch_liver_normal_unet --use_batch_norm 2>test_logs/256_batch_liver_normal_unet.log 1>test_logs/256_batch_liver_normal_unet.err) &

nohup $(python ./src/run_testing.py 64 2 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 128_batch_tumor_hard_tanh --use_batch_norm --tumor 2>test_logs/128_batch_tumor_hard_tanh.log 1>test_logs/128_batch_tumor_hard_tanh.err; \
        python ./src/run_testing.py 16 2 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 256_batch_tumor_hard_tanh --use_batch_norm --tumor 2>test_logs/256_batch_tumor_hard_tanh.log 1>test_logs/256_batch_tumor_hard_tanh.err; \
        python ./src/run_testing.py 64 2 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 128_batch_tumor_normal_unet --use_batch_norm --tumor 2>test_logs/128_batch_tumor_normal_unet.log 1>test_logs/128_batch_tumor_normal_unet.err;  \
        python ./src/run_testing.py 16 2 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 256_batch_tumor_normal_unet --use_batch_norm --tumor 2>test_logs/256_batch_tumor_normal_unet.log 1>test_logs/256_batch_tumor_normal_unet.err) &

nohup $(python ./src/run_testing.py 64 3 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 128_nobatch_liver_hard_tanh 2>test_logs/128_nobatch_liver_hard_tanh.log 1>test_logs/128_nobatch_liver_hard_tanh.err; \
        python ./src/run_testing.py 16 3 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 256_nobatch_liver_hard_tanh 2>test_logs/256_nobatch_liver_hard_tanh.log 1>test_logs/256_nobatch_liver_hard_tanh.err; \
        python ./src/run_testing.py 64 3 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 128_nobatch_liver_normal_unet 2>test_logs/128_nobatch_liver_normal_unet.log 1>test_logs/128_nobatch_liver_normal_unet.err;  \
        python ./src/run_testing.py 16 3 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 256_nobatch_liver_normal_unet 2>test_logs/256_nobatch_liver_normal_unet.log 1>test_logs/256_nobatch_liver_normal_unet.err) &

nohup $(python ./src/run_testing.py 64 4 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 128_nobatch_tumor_hard_tanh --tumor 2>test_logs/128_nobatch_tumor_hard_tanh.log 1>test_logs/128_nobatch_tumor_hard_tanh.err; \
        python ./src/run_testing.py 16 4 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "hard_tanh" 256_nobatch_tumor_hard_tanh --tumor 2>test_logs/256_nobatch_tumor_hard_tanh.log 1>test_logs/256_nobatch_tumor_hard_tanh.err; \
        python ./src/run_testing.py 64 4 128 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 128_nobatch_tumor_normal_unet --tumor 2>test_logs/128_nobatch_tumor_normal_unet.log 1>test_logs/128_nobatch_tumor_normal_unet.err;  \
        python ./src/run_testing.py 16 4 256 /home/macierz/s175573/tomography/lits_prepared/metadata.csv /home/macierz/s175573/tomography/lits_prepared "normal_unet" 256_nobatch_tumor_normal_unet --tumor 2>test_logs/256_nobatch_tumor_normal_unet.log 1>test_logs/256_nobatch_tumor_normal_unet.err) &
