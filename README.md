
# BM_MVP-Net: Multi-view FPN with Position-aware Attention for Deep Universal Lesion Detection

## Installation
This code is based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Please see it for installation.


## Environment
  - Python (Tested on 3.6)
  - PyTorch (Tested on 0.4.1.post2)

## Data preparation
Download DeepLesion dataset [here](https://nihcc.app.box.com/v/deeplesion).

We provide coco-style json annotation files converted from DeepLesion. Unzip Images_png.zip and make sure to put files as following sturcture:

```
data
  ├──DeepLesion
        ├── annotations
        │   ├── deeplesion_train.json
        │   ├── deeplesion_test.json
        │   ├── deeplesion_val.json
        └── Images_png
              └── Images_png
               │    ├── 000001_01_01
               │    ├── 000001_03_01
               │    ├── ...
```

## Training
To train MVP-Net with 9 slices model, run:
```
bash multi_windows_9_slices.sh train
```
We also provide our re-implementation of [3DCE](https://arxiv.org/pdf/1806.09648.pdf), see 3DCE_*.sh for training and testing.

## Testing
After training, put the model path into .sh file, after '--load_ckpt', and run:
```
bash multi_windows_9_slices.sh test
```




