# Multimodal Spiking Neural Network for Image-Text Retrieval

Our source code is built on top of the [VSE$\infty$](https://github.com/woodfrog/vse_infty), [USER](https://github.com/zhangy0822/USER) and [SpikingJelly](https://github.com/fangwei123456/spikingjelly)in PyTorch. 

## Environment
See the enviroment requirements in `requirements.txt`.
```
torch==2.1.2+cu118

spikingjelly==0.0.0.0.12

cupy-cuda11x==12.3.0
```

## Data
We organize all data used in the experiments in the same manner as [vse_inf](https://github.com/woodfrog/vse_infty).

The pre-computed region dataset extracted by FasterR-CNN is available in the README file of [vse_inf](https://github.com/woodfrog/vse_infty). You can download it from [here](https://www.dropbox.com/scl/fo/vmd0dvz20t7aae9jal0nc/ALoI0grReuah2PB5NgHGmac?rlkey=rei5ljf7hro7chkxltkcs0odr&e=1&dl=0).

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

Modify the corresponding arguments and run `train_region_coco.sh` or `train_region_f30k.sh`.

## Evaluation
Modify the corresponding arguments and run `test_region_coco.sh` or `test_region_f30k.sh`.


