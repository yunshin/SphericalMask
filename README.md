[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spherical-mask-coarse-to-fine-3d-point-cloud/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=spherical-mask-coarse-to-fine-3d-point-cloud)

## [CVPR 2024] Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation 

<a href="https://arxiv.org/abs/2312.11269"><img src=docs/sph_mask.jpeg></a>
[Sangyun Shin](https://www.cs.ox.ac.uk/people/sangyun.shin/),
[Kaichen Zhou](https://www.cs.ox.ac.uk/people/kaichen.zhou/),
[Madhu Vankadari](https://madhubabuv.github.io/),
[Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/),
[Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/)<br>
Cyber-Physical Systems Group, Department of Computer Science, University of Oxford
> **Abstract**: 
Coarse-to-fine 3D instance segmentation methods show weak performances compared to recent Grouping-based, Kernel-based and Transformer-based methods. We argue that this is due to two limitations: 1) Instance size overestimation by axis-aligned bounding box(AABB) 2) False negative error accumulation from inaccurate box to the refinement phase. In this work, we introduce Spherical Mask, a novel coarse-to-fine approach based on spherical representation, overcoming those two limitations with several benefits. Specifically, our coarse detection estimates each instance with a 3D polygon using a center and radial distance predictions, which avoids excessive size estimation of AABB. To cut the error propagation in the existing coarse-to-fine approaches, we virtually migrate points based on the polygon, allowing all foreground points, including false negatives, to be refined. 
During inference, the proposal and point migration modules run in parallel and are assembled to form binary masks of instances. We also introduce two margin-based losses for the point migration to enforce corrections for the false positives/negatives and cohesion of foreground points, significantly improving the performance. Experimental results from three datasets, such as ScanNetV2, S3DIS, and STPLS3D, show that our proposed method outperforms existing works, demonstrating the effectiveness of the new instance representation with spherical coordinates.


Details of the model architecture and experimental results can be found in [our paper](https://arxiv.org/abs/2312.11269).

## Quick Demo :fire:

### [ScanNetv2](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap)

| Dataset | AP | AP_50 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|
| ScanNet val | 62.6 | 81.9 | [config](configs/scannetv2/spherical_mask.yaml) | [checkpoint](https://drive.google.com/file/d/1WJtBr3nxaCaGCA_z1_dpu9bISnPAoxoL/view?usp=drive_link)

For the best training result, we recommend initializing the encoder with the pretrained-weights checkpoint([Download](https://drive.google.com/file/d/1TXGV-lVmmw94AJkqo6_Ms8wO5aVKhFdz/view?usp=drive_link)) from [ISBNet](https://arxiv.org/abs/2303.00246). 
After downloading the pre-trained weights, please specify the path in configs/scannetv2/spherical_mask.yaml
```shell
# train 
python python tools/train.py configs/scannetv2/spherical_mask.yaml --trainall --exp_name defaults
# test
python python tools/test.py configs/scannetv2/spherical_mask.yaml --ckpt path_to_ckpt.pth
```
More detailed instructions on library dependencies and environments will be uploaded soon.

**Please CITE** our paper if you found this repository helpful for producing publishable results or incorporating it into other software.
```bibtext
@inproceedings{shin2024spherical,
 author={Sangyun Shin, Kaichen Zhou, Madhu Vankadari, Andrew Markham, Niki Trigoni},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 title={Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation},
 year= {2024}
}
```

## Datasets :floppy_disk:

- [x] ScanNetV2

## Acknowledgements :clap:
This repo is built upon [ISBNet](https://github.com/VinAIResearch/ISBNet), [SpConv](https://github.com/traveller59/spconv), [DyCo3D](https://github.com/aim-uofa/DyCo3D), [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet), and [SoftGroup](https://github.com/thangvubk/SoftGroup). 

## Contacts :email:
If you have any questions or suggestions about this repo, please feel free to contact me (kimshin812@gmail.com or sangyun.shin@cs.ox.ac.uk).
