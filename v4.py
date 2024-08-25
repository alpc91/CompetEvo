import os
import subprocess


sds = [1,8,24]
gpu_index = 2

for sd in sds:
    #python train.py --cfg config/turn-to-goal-evoant-v1.yaml --num_threads 224 --gpu_index 2
    command = f"python train.py --cfg config/turn-to-goal-evoant-v4.yaml --num_threads 224 --gpu_index {gpu_index} --seed {sd}"

    os.system(command)
