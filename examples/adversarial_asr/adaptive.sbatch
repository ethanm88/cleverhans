#!/bin/bash
#SBATCH --time=00:00:10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3


python2.7 -m pip install librosa
python2.7 -m pip install Cython
python2.7 -m pip install pyroomacoustics
python2.7 -m pip install pydub

python2.7 generate_adaptive_adv.py

