# NARF
Neural Articulated Radiance Field

Atsuhiro Noguchi, Xiao Sun, Stephen Lin, Tatsuya Harada

[[Paper]](https://arxiv.org/abs/2104.03110) [[Code]](https://github.com/nogu-atsu/NARF/tree/main/src)

## Abstract
We present Neural Articulated Radiance Field (NARF), a novel deformable 3D representation for articulated objects learned from images. 
NARF extends Neural Radiance Fields (NeRF), which can only represent a single static scene, to articulated objects. 
In formulating an implicit representation of 3D articulated objects, our method considers only the rigid transformation of the most relevant object part in solving for the radiance field at each 3D location. In this way, the proposed method represents pose-dependent changes without significantly increasing the computational complexity. NARF is fully differentiable and can be trained from images with pose annotations. Moreover, through the use of an autoencoder, it can learn appearance variations over multiple instances of an object class. Experiments show that the proposed method is efficient and can generalize well to novel poses.

## Method
We extend Neural Radiance Fields (NeRF) to articulated objects.
NARF is a NeRF conditioned on skeletal parameters and skeletal posture, and is an MLP that outputs the density and color of a point with 3D position and 2D viewing direction as input.
Since articulated objects can be regarded as multiple rigid bodies connected by joints, the following two assumptions can be made

- The density of each part does not change in the coordinate system fixed to the part.
- A point on the surface of the object belongs to only one of the parts.

Therefore, we transform the input 3D coordinates into local coordinates of each part and use them as input for the model. From the second hypothesis, we use selector MLP to select only one necessary coordinate and mask the others.

An overview of the model is shown in the figure.

![overview](https://github.com/nogu-atsu/NARF/wiki/images/overview.jpg)

## Results
The proposed NARF is capable of rendering images with explicit control of the viwepoint, bone pose, and bone parameters. These representations are disentangled and can be controlled independently.

Viewpoint change (seen in training)

<img src="https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_camera_arf.mp4.mp4.gif" width="640px">

Pose change (unseen in training)

<img src="https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_pose_arf.mp4.gif" width="640px">

Bone length change (unseen in training)

<img src="https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_bone_arf.mp4.mp4.gif" width="640px">

NARF generalizes well to unseen viewpoints during training.

<img src="https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_camera_ood_arf.mp4.mp4.gif" width="640px">

Furthermore, NARF can render segmentation for each part by visualizing the output values of the selector.

<img src="https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_camera_arf_segmentation.mp4.mp4.gif" width="640px">

NARF can learn appearance variations by combining with autoencoder. 
The video below visualizes the disentangled representations and segmentation masks learned by NARF autoencoder.

<img src="https://github.com/nogu-atsu/NARF/wiki/images/ae_results.mp4.gif" width="1024px">


## BibTex
```
@article{2021narf
  author    = {Noguchi, Atsuhiro and Sun, Xiao and Lin, Stephen and Tatsuya, Harada},
  title     = {Neural Articulated Radiance Field},
  journal   = {arXiv preprint arXiv:2104.03110},
  year      = {2021},
}
```
