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

time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_eval_train-17.20  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5 

time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_016.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_017.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_018.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_019.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_020.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_021.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_022.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_023.pth.tar --gau_sigma 5.5
time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_eval_train-17.20/epoch_024.pth.tar --gau_sigma 5.5
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.5_0.5_sigma_5.5_act_weight_5  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5 --action_weight_ratio 5

# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_016.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_017.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_018.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_019.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_020.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_021.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_022.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_023.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_5/epoch_024.pth.tar --gau_sigma 5.5

# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.5_0.5_sigma_5.5_act_weight_10_2  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5 --action_weight_ratio 10

# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_016.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_017.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_018.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_019.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_020.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_021.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_022.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_023.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_10_2/epoch_024.pth.tar --gau_sigma 5.5

# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.5_0.5_sigma_5.5_act_weight_100  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5 --action_weight_ratio 100

# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_016.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_017.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_018.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_019.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_020.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_021.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_022.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_023.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_act_weight_100/epoch_024.pth.tar --gau_sigma 5.5


# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_2/epoch_022.pth.tar --gau_sigma 5.5
# time python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce_0.5_0.5_sigma_5.5_2/epoch_024.pth.tar --gau_sigma 5.5
#time python ./eval.py ./configs/epic_slowfast.yaml /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_three_heads2/ckpt/epic_slowfast_reproduce_0.3_0.7_sigma_5.5/epoch_020.pth.tar --gau_sigma 5.5
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_sigma_5.5  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.5_0.5_sigma_5.5  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5  --noun_cls_weight 0.5
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.4_0.6_sigma_5.5  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.4  --noun_cls_weight 0.6

#time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_reproduce_0.2_0.8_resum_n_5.5_sigma/epoch_070.pth.tar

# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_resum_n_3.5_sigma  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 3.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_resum_n_4.5_sigma  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 4.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_resum_n_5.5_sigma  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_resum_n_6.5_sigma  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 6.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_resum_n_7.5_sigma  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 7.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# #time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_5.5 --resum ./ckpt/epic_slowfast_noun_reproduce_5.5/epoch_020.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5

#time python ./train.py ./configs/epic_slowfast.yaml --output weight_combine  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_three_heads_2/ckpt/weight_combine/weight_combine.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 1 --noun_cls_weight 1

# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.5_0.5_resum_n  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.2_0.8_resum_n  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.2 --noun_cls_weight 0.8
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.3_0.7_resum_n  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.3 --noun_cls_weight 0.7
# time python ./train.py ./configs/epic_slowfast.yaml --output reproduce_0.4_1.6_resum_n  --resum /mnt/storage/scratch/dm19329/C2-Action-Detection/EPIC_challenge/actionformer_release_noun/ckpt/epic_slowfast_noun_reproduce_5.5_22.49/epoch_025_22.49.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.4 --noun_cls_weight 1.6

# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce   --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5  --verb_cls_weight 1 --noun_cls_weight 1 
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5.5/epoch_025.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5


# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_6 --gau_sigma 6 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_6/epoch_020.pth.tar --gau_sigma 6 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_6/epoch_025.pth.tar --gau_sigma 6 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_6/epoch_030.pth.tar --gau_sigma 6 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_6/epoch_034.pth.tar --gau_sigma 6 --sigma1 0.5 --sigma2 0.5


# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_5.5 --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5.5/epoch_020.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5.5/epoch_025.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5.5/epoch_030.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5.5/epoch_034.pth.tar --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5

# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_5 --gau_sigma 5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5/epoch_020.pth.tar --gau_sigma 5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5/epoch_025.pth.tar --gau_sigma 5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5/epoch_030.pth.tar --gau_sigma 5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_5/epoch_034.pth.tar --gau_sigma 5 --sigma1 0.5 --sigma2 0.5

# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_4.5 --gau_sigma 4.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4.5/epoch_020.pth.tar --gau_sigma 4.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4.5/epoch_025.pth.tar --gau_sigma 4.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4.5/epoch_030.pth.tar --gau_sigma 4.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4.5/epoch_034.pth.tar --gau_sigma 4.5 --sigma1 0.5 --sigma2 0.5

# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_4 --gau_sigma 4 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4/epoch_020.pth.tar --gau_sigma 4 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4/epoch_025.pth.tar --gau_sigma 4 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4/epoch_030.pth.tar --gau_sigma 4 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_4/epoch_034.pth.tar --gau_sigma 4 --sigma1 0.5 --sigma2 0.5

# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_3.5 --gau_sigma 3.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3.5/epoch_020.pth.tar --gau_sigma 3.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3.5/epoch_025.pth.tar --gau_sigma 3.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3.5/epoch_030.pth.tar --gau_sigma 3.5 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3.5/epoch_034.pth.tar --gau_sigma 3.5 --sigma1 0.5 --sigma2 0.5

# time python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce_3 --gau_sigma 3 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3/epoch_020.pth.tar --gau_sigma 3 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3/epoch_025.pth.tar --gau_sigma 3 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3/epoch_030.pth.tar --gau_sigma 3 --sigma1 0.5 --sigma2 0.5
# time python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce_3/epoch_034.pth.tar --gau_sigma 3 --sigma1 0.5 --sigma2 0.5
