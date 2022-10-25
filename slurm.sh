#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=voting
#SBATCH --partition gpu
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00


module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch

time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_eval_train-17.20  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5 

time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/name_of_best_model --gau_sigma 5.5
