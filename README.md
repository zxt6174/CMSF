# SiFu: Cross Modal Image-Text Retrieval via Spike Fusion

Our source code of SiFu accepted by TIP will be released as soon as possible. It is built on top of the [vse_inf](https://github.com/woodfrog/vse_infty) in PyTorch. 
## Data
We organize all data used in the experiments in the same manner as [vse_inf](https://github.com/woodfrog/vse_infty):

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── images   # raw coco images
│   │      ├── train2014
│   │      └── val2014
│   │
│   └── id_mapping.json  # mapping from coco-id to image's file name
│   
│
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── flickr30k-images   # raw coco images
│   │      ├── xxx.jpg
│   │      └── ...
│   └── id_mapping.json  # mapping from f30k index to image's file name
│
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```

## Training
Train MSCOCO and Flickr30K from scratch:

Modify the corresponding arguments and run `train_region_coco.sh` or `train_region_f30k.sh`

## Evaluation
Modify the corresponding arguments in `eval.py` and run `python eval.py`.


