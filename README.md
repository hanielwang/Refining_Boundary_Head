# Refining Boundary Head

This repository implements the boundaries head proposed in the paper:

Hanyuan Wang, Majid Mirmehdi, Dima Damen, Toby Perrett, **Refining Action Boundaries for One-stage Detection**, *AVSS*, 2022

[[arXiv paper]]( )

This repository is based on [ActionFormer](https://github.com/happyharrycn/actionformer_release).



## Citing

When using this code, kindly reference:

```

```

## Dependencies

* Python 3.5+
* PyTorch 1.11
* CUDA 11.0+
* GCC 4.9+
* TensorBoard
* NumPy 1.11+
* PyYaml
* Pandas
* h5py
* joblib

Complie NMS code by: 
```
cd ./libs/utils
python setup.py install --user
cd ../..
```


## Preparation

### Datasets and feature

You can download the annotation repository of EPIC-KITCHENS-100 at [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations)). Place it into a folder: ./data/epic_kitchens/annotations.

You can download the videos of EPIC-KITCHENS-100 at [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts).

You can download the feature on THUMOS14 at here [here](https://uob-my.sharepoint.com/:u:/g/personal/dm19329_bristol_ac_uk/EeXBKfXuurxNiZ3wazARQQsBD7j76jQMknSTgUTmXFYOog?e=Nt10i2). Place it into a folder: ./data/epic_kitchens/features.

If everything goes well, you can get the folder architecture of ./data like this:

    data                       
    └── epic_kitchens                    
        ├── features              
        └── annotations


### Pretrained models

You can download our pretrained models on EPIC-KITCHENS-100 at [here](https://uob-my.sharepoint.com/:u:/g/personal/dm19329_bristol_ac_uk/Ee4E66e04lhPsajKJZL-bHcBkVEGzkP-A8HjjALAUAPOEQ?e=IwiDkx).



## Training/validation on EPIC-KITCHENS-100
To train the model run:
```
python ./train.py ./configs/epic_slowfast.yaml --output reproduce  --gau_sigma 5.5 --sigma1 0.5 --sigma2 0.5 --verb_cls_weight 0.5 --noun_cls_weight 0.5 

```

To validate the model run:
```
python ./eval.py ./configs/epic_slowfast.yaml ./ckpt/epic_slowfast_reproduce/name_of_the_best_model --gau_sigma 5.5
```

## Results
```
[RESULTS] Action detection results_self.ap_action

|tIoU = 0.10: mAP = 19.19 (%)
|tIoU = 0.20: mAP = 18.61 (%)
|tIoU = 0.30: mAP = 17.47 (%)
|tIoU = 0.40: mAP = 16.30 (%)
|tIoU = 0.50: mAP = 14.33 (%)
Avearge mAP: 17.18 (%)
[RESULTS] Action detection results_self.ap_noun

|tIoU = 0.10: mAP = 23.58 (%)
|tIoU = 0.20: mAP = 22.40 (%)
|tIoU = 0.30: mAP = 21.03 (%)
|tIoU = 0.40: mAP = 19.27 (%)
|tIoU = 0.50: mAP = 16.39 (%)
Avearge mAP: 20.53 (%)
[RESULTS] Action detection results_self.ap_verb

|tIoU = 0.10: mAP = 23.75 (%)
|tIoU = 0.20: mAP = 22.68 (%)
|tIoU = 0.30: mAP = 21.22 (%)
|tIoU = 0.40: mAP = 19.19 (%)
|tIoU = 0.50: mAP = 16.73 (%)
Avearge mAP: 20.71 (%)
```

## Reference

This implementation is based on [ActionFormer](https://github.com/happyharrycn/actionformer_release).

Our main contribution is in: 
```
./libs/modeling/meta_archs.:
* We incorporate the estimation of boundary confidence into prediction heads. 
* We merged the classification heads of verb and noun, so the model can predict results for action task. 
* We implemented label assignment for boundary confidence.

./libs/modeling/losses.py:
* We added the supervision of boundary confidence, including confidence scaling and loss function calculation.

libs/datasets/epic_kitchens.py:
* We Loaded data for noun and verb together. 

/libs/utils/nms.py:
* We changed the sort of NMS through action socres instead of separate verb/noun scores.
```