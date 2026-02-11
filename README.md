# ADGaussian: Generalizable Gaussian Splatting for Autonomous Driving via Multi-modal Joint Learning [ICRA 2026]

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2504.00437)

## Introduction

This repository is an official implementation of [ADGaussian](https://maggiesong7.github.io/research/ADGaussian/)

<div align="center">
  <img src="figs/overview.png"/>
</div><br/>


Given monocular posed image with sparse depth as input, ADGaussian first extracts well-fused multi-modal features through Multi-modal Feature Matching, which contains a siamese-style encoder and a cross-attention decoder enhanced by Depth-guided positional embedding (DPE). Subsequently, the Gaussian Head and Geometry Head, augmented with Multi-scale Gaussian Decoding, are utilized to predict different Gaussian parameters. SDTR got accepted by ICRA 2026.


## Preparation

#### Environment

This implementation is built upon [MVSplat](https://github.com/donydchen/mvsplat).

```bash
pip install scripts/submodules/diff-gaussian-rasterization_ms
```

#### Pretrained weights   

We provide the pre-trained weights of our ADGaussian at [google drive](https://drive.google.com/drive/folders/1vlPobBvzWmfVmdkNMrioH9rNF475QxpM?usp=sharing).


## Train & inference

You can train the model following:

```bash
python -m src.main +experiment=waymo data_loader.train.batch_size=1
```

You can evaluate the model following:

```bash
python -m src.main +experiment=waymo checkpointing.load=[ckpt_path] mode=test 
```

## Main Results

|        scene        |  PSNR  |  SSIM  |  LPIPS |
| :-----------------: | :----: | :----: | :----: |
| static32-003        | 31.09  | 0.931  | 0.059  |
| static32-069        | 31.17  | 0.923  | 0.073  |
| static32-232        | 30.52  | 0.904  | 0.083  |
| static32-495        | 31.21  | 0.929  | 0.056  |


## Acknowledgement

Many thanks to the authors of [MVSplat](https://github.com/donydchen/mvsplat) and [MAST3R](https://github.com/naver/mast3r) .


## Citation

If you find this project useful for your research, please consider citing: 

```bibtex   
@article{song2025adgaussian,
  title={ADGaussian: Generalizable Gaussian Splatting for Autonomous Driving with Multi-modal Inputs},
  author={Song, Qi and Li, Chenghong and Lin, Haotong and Peng, Sida and Huang, Rui},
  journal={arXiv preprint arXiv:2504.00437},
  year={2025}
}
```

## Contact

If you have any questions, feel free to open an issue or contact us at qisong@link.cuhk.edu.cn.
