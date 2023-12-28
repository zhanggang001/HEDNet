# HEDNet

It is the official code release of [HEDNet](https://arxiv.org/pdf/2310.20234.pdf), which achieves state-of-the-art performance on large-scale Waymo Open Dataset.


## Changelog
[2023-12-25] **NEW:** Initial code release.


### Results on Waymo Open

We implemented HEDNet on Waymo Open based on OpenPCDet.

#### Validation set
| Model | L1 mAP/mAPH | L2 mAP/mAPH | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|-------------------------------------------------------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [HEDNet](https://github.com/zhanggang001/HEDNet/blob/master/tools/cfgs/hednet_models/hednet_8x_1f_onestage_D1_2x.yaml)                             | 81.4/79.5 | 75.3/73.4 | 81.1/80.6 | 73.2/72.7 | 84.4/80.0 | 76.8/72.6 | 78.7/77.7 | 75.8/74.9 |

### Test set
| Model | L1 mAP/mAPH | L2 mAP/mAPH | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|-------------------------------------------------------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [HEDNet](https://github.com/zhanggang001/HEDNet/blob/master/tools/cfgs/hednet_models/hednet_8x_1f_onestage_D1_2x.yaml)                             | 82.2/80.2 | 76.9/75.0 | 84.2/83.8 | 77.0/76.6 | 84.1/79.7 | 78.3/74.0 | 78.2/77.0 | 75.4/74.3 |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/).


### Results on NuScenes
We implemented HEDNet on NuScenes based on mmdetection3d, because the TransFusion-L implemented on OpenPCDet achieved lower accuracy than on mmdetection3d. We will unify the code in the future.

#### Validation set
|Model|   mATE |  mASE  |  mAOE  | mAVE  | mAAE  |  mAP  |  NDS   |                                              download                                              |
|----------------------------------------------------------------------------------------------------|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:--------------------------------------------------------------------------------------------------:|
| [HEDNet](https://github.com/zhanggang001/HEDNet-nusc/blob/master/configs/hednet/hednet_transfusion_L_nusc.py)                         | 27.5 | 25.1 | 26.3 |	23.3 | 18.7 | 67.0 | 71.4 | [ckpt](https://cloud.tsinghua.edu.cn/f/40f6d51e038f4c158616/?dl=1) |

#### Test set
|Model| mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | download |
|---|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:----:|
| [HEDNet](https://github.com/zhanggang001/HEDNet-nusc/blob/master/configs/hednet/hednet_transfusion_L_nusc_trainval.py) | 25.0 | 23.8 | 31.7 | 24.0 | 13.0 | 67.5 | 72.0 | [json](https://cloud.tsinghua.edu.cn/f/bf54afa8d28c4d74affe/?dl=1) |

## Installation and usage

For `OpenPCDet`, please refer to [INSTALL.md](docs/INSTALL.md) and [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the installation and usage, respectively. We used python 3.8, pytorch 1.10, cuda11.3, spconv-cu113 2.3.3.

For HEDNet on nuScenes, we release the code in another repository, please refer to [HEDNet-nusc](https://github.com/zhanggang001/HEDNet-nusc).


## Citation
```
@inproceedings{
  zhang2023hednet,
  title={{HEDN}et: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds},
  author={Gang Zhang and Chen Junnan and Guohuan Gao and Jianmin Li and Xiaolin Hu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
}
```

## Acknowleadgement
This work was supported in part by the National Key Research and Development Program of China (No. 2021ZD0200301) and the National Natural Science Foundation of China (Nos. U19B2034, 61836014) and THU-Bosch JCML center.
