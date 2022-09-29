#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=voting
#SBATCH --partition gpu
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00


# source activate
# # # source deactivate
# conda deactivate
# # conda init bash
#conda activate why
# export MASTER_ADDR=localhost
# export MASTER_PORT=5678


module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/EPIC_17.20.pth.tar --gau_sigma 5.5



