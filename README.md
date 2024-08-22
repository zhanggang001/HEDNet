# HEDNet & SAFDNet

It is the official code release of [HEDNet (NeurIPS 2023)](https://arxiv.org/pdf/2310.20234.pdf) and [SAFDNet (CVPR 2024)](https://arxiv.org/abs/2403.05817).
We implemented HEDNet and SAFDNet on all datasets based on OpenPCDet.

### Results on Waymo Open

#### Validation set
| Model | mAP/mAPH_L1 | mAP/mAPH_L2 | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|-------------------------------------------------------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [HEDNet-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_1f_1x_waymo.yaml)    | 81.1/79.1 | 75.0/73.0 | 80.2/79.7 | 72.3/71.9 | 79.3/76.4 | 76.4/71.9 | 79.1/78.1 | 76.2/75.3 |
| [SAFDNet-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_1x_waymo.yaml) | 81.2/79.2 | 75.1/73.2 | 80.2/79.7 | 72.2/71.8 | 79.9/76.9 | 76.8/72.6 | 79.1/78.1 | 76.2/75.2 |
| [HEDNet-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_1f_2x_waymo.yaml)    | 81.4/79.5 | 75.3/73.4 | 81.1/80.6 | 73.2/72.7 | 84.4/80.0 | 76.8/72.6 | 78.7/77.7 | 75.8/74.9 |
| [SAFDNet-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_2x_waymo.yaml) | 81.8/79.9 | 75.7/73.9 | 80.6/80.1 | 72.7/72.3 | 84.7/80.4 | 77.3/73.1 | 80.0/79.0 | 77.2/76.2 |

### Test set
| Model | mAP/mAPH_L1 | mAP/mAPH_L2 | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Leaderboard |
|-------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| HEDNet-2x | 82.2/80.2 | 76.9/75.0 | 84.2/83.8 | 77.0/76.6 | 84.1/79.7 | 78.3/74.0 | 78.2/77.0 | 75.4/74.3 | [link](https://waymo.com/open/challenges/detection-3d/results/6a0c9b2c-7ae6/1694521910913280/)
| SAFDNet-2x | 81.9/79.8 | 76.5/74.6 | 83.9/83.5 | 76.6/76.2 | 84.3/79.8 | 78.3/74.1 | 77.5/76.3 | 74.6/73.4 | Todo |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/).


### Results on NuScenes

#### Validation set
|Model|   mATE |  mASE  |  mAOE  | mAVE  | mAAE  |  mAP  |  NDS   |  Checkpoint |
|----------------------------------------------------------------------------------------------------|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:--------------------------------------------------------------------------------------------------:|
| [HEDNet](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_20e_nusences.yaml)  | 26.9 | 24.9 | 26.3 |	25.5 | 18.0 | 67.3 | 71.5 | [ckpt](https://cloud.tsinghua.edu.cn/f/4c1880be8453468baa4c/?dl=1) & [log](https://cloud.tsinghua.edu.cn/f/d31fb828452449a49700/?dl=1)|
| [SAFDNet](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_20e_nuscenes.yaml)        | 27.3 | 25.4 | 27.6 |	25.5 | 18.3 | 66.5 | 70.9 | [ckpt](https://cloud.tsinghua.edu.cn/f/5a327be6f8a440fd9d22/?dl=1) & [log](https://cloud.tsinghua.edu.cn/f/25c72526e9ad4f95851b/?dl=1) |

#### Test set
|Model| mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | Leaderboard |
|---|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:----:|
| HEDNet | 25.0 | 23.8 | 31.7 | 24.0 | 13.0 | 67.5 | 72.0 | [json](https://cloud.tsinghua.edu.cn/f/afc2275bdbef4ae4ac11/?dl=1) |
| SAFDNet | 25.1 | 24.2 | 31.1 | 25.8 | 12.7 | 68.3 | 72.3 | [json](https://cloud.tsinghua.edu.cn/f/6b254feee21445318e0f/?dl=1) |


### Results on Argoverse2

#### Validation set
|Model| mAP | Checkpoint |
|----|-------:|:------:|
| [HEDNet-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_1f_1x_argo.yaml)     | 37.3 | Todo |
| [SAFDNet-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_1x_argo.yaml)  | 39.4 | Todo |


## Installation and usage

For `OpenPCDet`, please refer to [INSTALL.md](docs/INSTALL.md) and [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the installation and usage, respectively. We used python 3.8, pytorch 1.10, cuda11.3, spconv-cu113 2.3.3. We provide a list of Python packages output from `pip freeze` [here](https://cloud.tsinghua.edu.cn/f/066bd16c12314eb6bc26/?dl=1), to help configure the environment.

You can create an experiment folder in any location, and organize it like this:
```
FOLDER_NAME:
├── ${PATH_TO_HEDNet_ROOT}/HEDNet/tools/cfgs
├── ${PATH_TO_HEDNet_ROOT}/HEDNet/data
├── xxx.yaml (copy the yaml file here)
├── dist_train.sh (copy the training script from tools/scripts here)
├── dist_test.sh (copy the testing script from tools/scripts here)
```
Then you can train and test models like this:
```
# Train with 8 gpus
./dist_train.sh xxx.yaml 8

# Test with 8 gpus
./dist_test.sh xxx.yaml 8 output/ckpt/xxx.pth
```

## TODO
- Since we rebuilt and unified the codebase for all datasets, the model accuracy of HEDNet and SAFDNet is slightly lower than the results released in the paper (by at most 0.3\% L2 mAPH on Waymo Open). You can run the previous branch ``HEDNet`` to get better results. We are trying to fix the gap and will update the code as soon as possible.
- Release the model checkpoints on nuScenes and Argoverse2 datasets.
- A cross-modal fusion method based on HEDNet and SAFDNet is on the way.


## Citation
```
@inproceedings{
  zhang2023hednet,
  title={{HEDNet}: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds},
  author={Gang Zhang and Chen Junnan and Guohuan Gao and Jianmin Li and Xiaolin Hu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
}

@misc{
  zhang2024safdnet,
  title={{SAFDNet}: A Simple and Effective Network for Fully Sparse 3D Object Detection},
  author={Gang Zhang and Junnan Chen and Guohuan Gao and Jianmin Li and Si Liu and Xiaolin Hu},
  year={2024},
  eprint={2403.05817},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Acknowleadgement
This two works were supported by the National Key Research and Development Program of China (No. 2021ZD0200301) and the National Natural Science Foundation of China (Nos. U19B2034, 61836014) and THU-Bosch JCML center.
