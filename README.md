# [ICCV2023] Explicit Motion Disentangling for Efficient Optical Flow Estimation

<p align="center">
Changxing Deng<sup>1</sup>, Ao Luo<sup>2</sup>, Haibin Huang<sup>3</sup>, Shaodan Ma<sup>1</sup>, Jiangyu Liu<sup>2</sup>, Shuaicheng Liu<sup>4,2</sup>
</p>
<p align="center">1. University of Macau, 2. Megvii Technology, 3. Kuaishou Technology, </p>
<p align="center">4. University of Electronic Science and Technology of China</p>

This repository provides the implementation for [Explicit Motion Disentangling for Efficient Optical Flow Estimation]()

# Abstract

In this paper, we propose a novel framework for optical flow estimation that achieves a good balance between performance and efficiency. Our approach involves disentangling global motion learning from local flow estimation, treating global matching and local refinement as separate stages. We offer two key insights: First, the multi-scale 4D cost-volume based recurrent flow decoder is computationally expensive and unnecessary for handling small displacement. With the separation, we can utilize lightweight methods for both parts and maintain similar performance. Second, a dense and robust global matching is essential for both flow initialization as well as stable and fast convergence for the refinement stage.
Towards this end, we introduce EMD-Flow, a framework that explicitly separates global motion estimation from the recurrent refinement stage. We propose two novel modules: Multi-scale Motion Aggregation (MMA) and Confidence-induced Flow Propagation (CFP). These modules leverage cross-scale matching prior and self-contained confidence maps to handle the ambiguities of dense matching in a global manner, generating a dense initial flow. Additionally, a lightweight decoding module is followed to handle small displacements, resulting in an efficient yet robust flow estimation framework. 
We further conduct comprehensive experiments on standard optical flow benchmarks with the proposed framework, and the experimental results demonstrate its superior balance between performance and runtime.

# Requirements
pytorch==1.10.2 \
torchvision==0.11.3 \
numpy==1.19.2 \
timm==0.4.12 \
tensorboard==2.6.0 \
scipy==1.5.2 \
pillow==8.4.0 \
opencv-python==4.5.5.64 \
cudatoolkit==11.3.1

# Evaluate
1. The weights of models are available on [Google Drive](https://drive.google.com/drive/folders/1GaotDD2PqQAbIgvS1TwJJbdabRqGp21J?usp=drive_link), and put the files into the folder **weights**.
2. Download the [Sintel](http://sintel.is.tue.mpg.de/) and [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) datasets and put them into the folder **data**
3. Evaluate our models by ` sh evaluate.sh `

# Citation

# Acknowledgement
The main framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official). We thank the authors for the contribution.
