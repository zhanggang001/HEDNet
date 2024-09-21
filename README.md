# HEDNet & SAFDNet

It is the official code release of [HEDNet (NeurIPS 2023)](https://arxiv.org/pdf/2310.20234.pdf) and [SAFDNet (CVPR 2024)](https://arxiv.org/abs/2403.05817).
We unify the codebase for HEDNet and SAFDNet on all datasets based on OpenPCDet.
Please note that, since we rebuilt the code, the results are slightly different from those in the original paper.

### Results on Waymo Open

#### Validation set
| Model | mAP/mAPH_L1 | mAP/mAPH_L2 | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|-------------------------------------------------------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [HEDNet-1f-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_1f_1x_waymo.yaml)    | 81.1/79.1 | 75.0/73.0 | 80.2/79.7 | 72.3/71.9 | 79.3/76.4 | 76.4/71.9 | 79.1/78.1 | 76.2/75.3 |
| [SAFDNet-1f-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_1x_waymo.yaml) | 81.2/79.2 | 75.1/73.2 | 80.2/79.7 | 72.2/71.8 | 79.9/76.9 | 76.8/72.6 | 79.1/78.1 | 76.2/75.2 |
| [HEDNet-1f-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_1f_2x_waymo.yaml)    | 81.4/79.5 | 75.3/73.4 | 81.1/80.6 | 73.2/72.7 | 84.4/80.0 | 76.8/72.6 | 78.7/77.7 | 75.8/74.9 |
| [SAFDNet-1f-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_2x_waymo.yaml) | 81.6/79.7 | 75.5/73.7 | 80.7/80.3 | 72.8/72.4 | 84.8/80.4 | 77.3/73.0 | 79.4/78.4 | 76.6/75.6 |
| [HEDNet-4f-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_4f_2x_waymo.yaml)    | 83.3/82.1 | 77.8/76.6 | 82.6/82.1 | 75.3/74.8 | 86.2/83.6 | 79.1/76.6 | 81.2/80.4 | 79.0/78.2 |
| [SAFDNet-4f-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_4f_2x_waymo.yaml) | 84.1/82.8 | 78.6/77.3 | 82.6/82.1 | 75.1/74.7 | 86.7/84.1 | 80.0/77.4 | 83.0/82.1 | 80.7/79.9 |

### Test set
| Model | mAP/mAPH_L1 | mAP/mAPH_L2 | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Submission |
|-------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| HEDNet-1f-2x | 82.2/80.2 | 76.9/75.0 | 84.2/83.8 | 77.0/76.6 | 84.1/79.7 | 78.3/74.0 | 78.2/77.0 | 75.4/74.3 | [link](https://waymo.com/open/challenges/detection-3d/results/6a0c9b2c-7ae6/1694521910913280/)
| SAFDNet-1f-2x | 81.9/79.8 | 76.5/74.6 | 83.9/83.5 | 76.6/76.2 | 84.3/79.8 | 78.3/74.1 | 77.5/76.3 | 74.6/73.4 | [link](https://waymo.com/open/challenges/detection-3d/results/ae022fcb-9223/1700186617972208/) |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/).


### Results on NuScenes

#### Validation set
|Model|   mATE |  mASE  |  mAOE  | mAVE  | mAAE  |  mAP  |  NDS   |  Checkpoint |
|----------------------------------------------------------------------------------------------------|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:--------------------------------------------------------------------------------------------------:|
| [HEDNet](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/hednet_models/hednet_20e_nusences.yaml)  | 27.2 | 25.1 | 26.5 |	25.7 | 18.0 | 67.1 | 71.3 | [ckpt](https://cloud.tsinghua.edu.cn/f/02fb0bb6e1f5443cba2d/?dl=1) & [log](https://cloud.tsinghua.edu.cn/f/8de86777f8974bd893b4/?dl=1)|
| [SAFDNet](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_20e_nuscenes.yaml)        | 27.3 | 25.3 | 27.1 |	25.2 | 18.3 | 66.6 | 71.0 | [ckpt](https://cloud.tsinghua.edu.cn/f/92e8977759ce41259980/?dl=1) & [log](https://cloud.tsinghua.edu.cn/f/143cdb06d6254dd7846e/?dl=1) |

#### Test set
|Model| mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | Submission |
|---|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:----:|
| HEDNet | 25.0 | 23.8 | 31.7 | 24.0 | 13.0 | 67.5 | 72.0 | [json](https://cloud.tsinghua.edu.cn/f/afc2275bdbef4ae4ac11/?dl=1) |
| SAFDNet | 25.1 | 24.2 | 31.1 | 25.8 | 12.7 | 68.3 | 72.3 | [json](https://cloud.tsinghua.edu.cn/f/6b254feee21445318e0f/?dl=1) |

**Note**: We originally implemented HEDNet on the nuScenes dataset using MMDetection3D. The TransFusion-L in OpenPCDet which uses the hierarchical 3D SECOND as its backbone achieves worse results than that in MMDetection3D. To unify the codebase, we release a single-stride 2D version for both HEDNet and SAFDNet on the nuScenes dataset. This version differs from the hierarchical 3D version described in the original paper but achieves similar results.

### Results on Argoverse2

#### Validation set
|Model| mAP | Checkpoint |
|----|-------:|:------:|
| [SAFDNet-1x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_1x_argo.yaml)  | 39.4 | [ckpt](https://cloud.tsinghua.edu.cn/f/505c0c8de13741bdbb78/?dl=1) & [log](https://cloud.tsinghua.edu.cn/f/76d9f936b2ec4032876d/?dl=1) |
| [SAFDNet-2x](https://github.com/zhanggang001/HEDNet/blob/main/tools/cfgs/safdnet_models/safdnet_1f_2x_argo.yaml)  | 39.9 | [ckpt](https://cloud.tsinghua.edu.cn/f/53ea58e6a5ee49f8ae85/?dl=1) & [log](https://cloud.tsinghua.edu.cn/f/7cf90924c006428db6b2/?dl=1) |

## Installation and usage

Please refer to [INSTALL.md](docs/INSTALL.md) and [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the installation and usage, respectively. We used python 3.8, pytorch 1.10, cuda11.3, spconv-cu113 2.3.3. We provide a list of Python packages output from `pip freeze` [here](https://cloud.tsinghua.edu.cn/f/066bd16c12314eb6bc26/?dl=1), to help configure the environment.

You can create an experiment folder in any location, and organize it like this:
```
FOLDER_NAME:
├── ${PATH_TO_HEDNet_ROOT}/HEDNet/tools/cfgs
├── ${PATH_TO_HEDNet_ROOT}/HEDNet/data
├── xxx.yaml (copy the yaml file here)
├── dist_train.sh (copy the training script from tools/scripts here)
├── dist_test.sh (copy the testing script from tools/scripts here)
```

For faster evaluation on the Waymo Open dataset, please download the [compute_detection_metrics_main](https://cloud.tsinghua.edu.cn/f/df1c6c0b629b4929a357/?dl=1) and [gt.bin](https://cloud.tsinghua.edu.cn/f/637df33ef3a44fd09fb8/?dl=1), and then put them under the HEDNet/data/waymo. You may need to excute `chmod +x compute_detection_metrics_main` to modify the file permission to make it an executable file. If you want to generate these two files on your own, please refer to the [this repo](https://github.com/Abyssaledge/faster-waymo-detection-evaluation/blob/master/docs/quick_start.md#local-compilation-without-docker-system-requirements).

Then you can train and test models like this:
```
# Train with 8 gpus
./dist_train.sh xxx.yaml 8

# Test with 8 gpus
./dist_test.sh xxx.yaml 8 output/ckpt/xxx.pth
```

## TODO
- A cross-modal fusion method based on HEDNet and SAFDNet is on the way.

<!-- ## FAQ -->

<!-- - Since we rebuilt and unified the codebase for all datasets, the model accuracy of HEDNet and SAFDNet is slightly lower than the results released in the paper (by at most 0.3\% L2 mAPH on Waymo Open). You can run the previous branch ``HEDNet`` to get better results. We are trying to fix the gap and will update the code as soon as possible. -->
<!-- - Release the model checkpoints on nuScenes and Argoverse2 datasets. -->

## Citation
```
@inproceedings{zhang2023hednet,
  title={{HEDNet}: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds},
  author={Zhang, Gang and Chen, Junnan and Gao, Guohuan and Li, Jianmin and Hu, Xiaolin},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023},
}

@inproceedings{zhang2024cvpr,
    title     = {{SAFDNet}: A Simple and Effective Network for Fully Sparse 3D Object Detection},
    author    = {Zhang, Gang and Chen, Junnan and Gao, Guohuan and Li, Jianmin and Liu, Si and Hu, Xiaolin},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {14477-14486}
}
```

## Acknowleadgement
This two works were supported by the National Key Research and Development Program of China (No. 2021ZD0200301) and the National Natural Science Foundation of China (Nos. U19B2034, 61836014) and THU-Bosch JCML center.
