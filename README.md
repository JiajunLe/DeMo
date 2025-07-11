# DeMo implementation

Pytorch implementation of DeMo for AAAI'25 paper ["DeMo: Deep Motion Field Consensus with Learnable Kernels for Two-view Correspondence Learning"](https://ojs.aaai.org/index.php/AAAI/article/view/32622), by [Yifan Lu](https://scholar.google.com/citations?user=h-9Ub_cAAAAJ&hl=zh-CN), [Jiajun Le](https://scholar.google.com/citations?user=uWhzrG4AAAAJ&hl=zh-CN&oi=sra), [Zizhuo Li](https://scholar.google.com/citations?user=bxuEALEAAAAJ&hl=zh-CN)
, [Yixuan Yuan](https://scholar.google.com/citations?user=Aho5Jv8AAAAJ&hl=zh-CN&oi=sra) and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ&hl).

This paper focuses on identifying reliable correspondences and estimating the relative pose between two images. We design a correspondence learning network called DeMo that for outlier rejection by capturing global motion consensus through consensus interpolation of the high-dimensional motion field generated by putative correspondences. It integrates regularization techniques within a Reproducing Kernel Hilbert Space (RKHS) to derive a concise interpolation formula, allowing for a closed-form solution. Additionally, a scene-adaptive sampling strategy is introduced to reduce computational complexity while maintaining accuracy.

This repo contains the code and data for essential matrix estimation described in our AAAI paper

If you find this project useful, please cite:

```
@inproceedings{Lu2025DeMo,
  title={DeMo: Deep Motion Field Consensus with Learnable Kernels for Two-view Correspondence Learning},
  author={Lu, Yifan and Le, Jiajun and Li, Zizhuo and Yuan, Yixuan and Ma, Jiayi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Requirements and Compilation

Please use Python 3.8.5, opencv-contrib-python (4.2.0.32) and Pytorch (>= 1.9.1). Other dependencies should be easily installed through pip or conda.

### Compile extra modules

```bash
cd network/knn_search
python setup.py build_ext --inplace
cd ../pointnet2_ext
python setup.py build_ext --inplace
```

## Example scripts

### Run the demo

You can run the feature matching for two images with DeMo.

```bash
cd ../../filter_demo && python filter.py
```

### Datasets and Pretrained models

Download the pretrained models from [here](https://drive.google.com/drive/folders/1dsDUVwEOMC0mExEPxTj3JjTw68fsUt34?usp=drive_link).

Download YFCC100M dataset.
```bash
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
tar -xvf raw_data_yfcc.tar.gz
```

Download SUN3D testing (1.1G) and training (31G) dataset if you need.
```bash
bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```

### Test pretrained model

We provide the models trained on the YFCC100M and SUN3D datasets, as detailed in our AAAI paper. By running the test script, you can obtain results similar to those presented in our paper. Note that the generated putative matches may differ if the data is regenerated.

```bash
cd ../test 
python test.py
```
To adjust the default settings for test, you can edit the `../test/config.py`.

### Train model on YFCC100M or SUN3D

After generating dataset for YFCC100M, run the tranining script.
```bash
cd ../network 
python main.py
```

## Acknowledgement
This code is borrowed from [LMCNet](https://github.com/liuyuan-pal/LMCNet.git) and [ConvMatch](https://github.com/SuhZhang/ConvMatch). If using the part of code related to data generation, testing and evaluation, please cite these papers.

```
@inproceedings{liu2021learnable,
  title={Learnable Motion Coherence for Correspondence Pruning},
  author={Liu, Yuan and Liu, Lingjie and Lin, Cheng and Dong, Zhen and Wang, Wenping},
  booktitle={CVPR}
  year={2021}
}
@inproceedings{zhang2023convmatch,
  title={ConvMatch: Rethinking Network Design for Two-View Correspondence Learning},
  author={Zhang, Shihua and Ma, Jiayi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
