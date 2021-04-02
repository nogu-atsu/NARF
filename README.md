# NARF
Neural Articulated Radiance Field

Atsuhiro Noguchi, Xiao Sun, Stephen Lin, Tatsuya Harada

[Paper]()

## Abstract
We present Neural Articulated Radiance Field (NARF), a novel deformable 3D representation for articulated objects learned from images. While recent advances in 3D implicit representation have made it possible to learn models of complex objects, learning pose-controllable representations of articulated objects remains a challenge, as current methods require 3D shape supervision and are unable to render appearance. In formulating an implicit representation of 3D articulated objects, our method considers only the rigid transformation of the most relevant object part in solving for the radiance field at each 3D location. In this way, the proposed method represents pose-dependent changes without significantly increasing the computational complexity. NARF is fully differentiable and can be trained from images with pose annotations. Moreover, through the use of an autoencoder, it can learn appearance variations over multiple instances of an object class. Experiments show that the proposed method is efficient and can generalize well to novel poses.

## Results
![pose](https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_camera_arf.mp4.mp4.gif)
![pose](https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_camera_arf_segmentation.mp4.mp4.gif)
![pose](https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_pose_arf.mp4.gif)
![pose](https://github.com/nogu-atsu/NARF/wiki/images/concat_inter_camera_ood_arf.mp4.mp4.gif)

## BibTex
```
@article{2021narf
  author    = {....},
  title     = {Neural Articulated Radiance Field},
  journal   = {arXiv preprint arXiv:????.?????},
  year      = {2020},
}
```
