# Plenoxels: Radiance Fields without Neural Networks

Alex Yu\*, Sara Fridovich-Keil\*, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa

UC Berkeley

Website and video: <https://alexyu.net/plenoxels>

arXiv: <https://arxiv.org/abs/2112.05131>

[Featured at Two Minute Papers YouTube](https://youtu.be/yptwRRpPEBM) 2022-01-11

Despite the name, it's not strictly intended to be a successor of svox

Citation:
```
@inproceedings{yu2022plenoxels,
      title={Plenoxels: Radiance Fields without Neural Networks}, 
      author={Sara Fridovich-Keil and Alex Yu and Matthew Tancik and Qinhong Chen and Benjamin Recht and Angjoo Kanazawa},
      year={2022},
      booktitle={CVPR},
}
```

## Setup
- Check ```conda_plenoxel_env.sh```

## Dataset preparation
- In order to run Plenoxel on ScanNet++ data, you first need to convert \<scene\>/nerfstudio/transforms.json into NSVF format [here](https://github.com/facebookresearch/NSVF?tab=readme-ov-file#prepare-your-own-dataset) by [VCAI-utils/ScanNetPPNSVF.py](https://github.com/Ryoto-Kato/VCAI-utils/blob/main/ScanNetPP2NSVF.py)

## Optimization
- change the data directory which contains all of scenes you want to train Plenoxel on

```
├── <scene name>
│   └── dslr
│       ├── nerfstudio
│       │   ├── transform.json
│       ├── nsvf
│       │   ├── bbox.txt
│       │   ├── edges.txt
│       │   ├── intrinsics.txt
│       │   ├── pose
│       │   │   ├── 0_DSC04849.txt
│       │   │   ├── 0_DSC04850.txt
│       │   │   ├── 0_DSC04851.txt
│       │   │   ├── 0_DSC04852.txt
│       │   │     ...
│       │   ├── rgb
│       │   │   ├── 0_DSC04849.JPG
│       │   │   ├── 0_DSC04850.JPG
│       │   │   ├── 0_DSC04851.JPG
│       │   │   ├── 0_DSC04852.JPG
│       │   │     ...
```

```sh
nohup python train_all_scenes.py
```
