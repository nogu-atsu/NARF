# Neural Articulated Radiance Field
# NARF


> Neural Articulated Radiance Field  
> Atsuhiro Noguchi, Xiao Sun, Stephen Lin, Tatsuya Harada  
> ICCV 2021  

[[Paper]](https://arxiv.org/abs/2104.03110) [[Code]](https://github.com/nogu-atsu/NARF_release#training)

## Abstract
We present Neural Articulated Radiance Field (NARF), a novel deformable 3D representation for articulated objects learned from images. While recent advances in 3D implicit representation have made it possible to learn models of complex objects, learning pose-controllable representations of articulated objects remains a challenge, as current methods require 3D shape supervision and are unable to render appearance. In formulating an implicit representation of 3D articulated objects, our method considers only the rigid transformation of the most relevant object part in solving for the radiance field at each 3D location. In this way, the proposed method represents pose-dependent changes without significantly increasing the computational complexity. NARF is fully differentiable and can be trained from images with pose annotations. Moreover, through the use of an autoencoder, it can learn appearance variations over multiple instances of an object class. Experiments show that the proposed method is efficient and can generalize well to novel poses.

## Method
We extend Neural Radiance Fields (NeRF) to articulated objects.
NARF is a NeRF conditioned on skeletal parameters and skeletal posture, and is an MLP that outputs the density and color of a point with 3D position and 2D viewing direction as input.
Since articulated objects can be regarded as multiple rigid bodies connected by joints, the following two assumptions can be made

- The density of each part does not change in the coordinate system fixed to the part.
- A point on the surface of the object belongs to only one of the parts.

Therefore, we transform the input 3D coordinates into local coordinates of each part and use them as input for the model. From the second hypothesis, we use selector MLP to select only one necessary coordinate and mask the others.

An overview of the model is shown in the figure.

![overview](https://github.com/nogu-atsu/NARF/wiki/images/overview.jpg)

The model is trained with the L2 loss between the generated image and the ground truth image.

## Results
The proposed NARF is capable of rendering images with explicit control of the viewpoint, bone pose, and bone parameters. These representations are disentangled and can be controlled independently.

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

NARF can learn appearance variations by combining it with an autoencoder. 
The video below visualizes the disentangled representations and segmentation masks learned by NARF autoencoder.

<img src="https://github.com/nogu-atsu/NARF/wiki/images/ae_results.mp4.gif" width="1024px">


## Training
## Envirionment
python 3.7.*
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardx pyyaml opencv-python pandas ninja easydict tqdm scipy scikit-image
```

## Dataset preparation
Dataset preparation code is based on https://github.com/ZhengZerong/DeepHuman

- Download THUman dataset from https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset and unzip it.
- Specify the path to `dataset_dir` in config.py
- update `data_list.txt` if necessary 
- Specify the path to save the data to `output_dir`
- Run `python render_THUman.py`

## Training
- Write config file like `configs/supervised_part_wise_NeRF/default.yml`. Do not change `default.yml`
    - `out_root`: root directory of output
    - `out`: output directory name
    - `data_root`: directory the `dataset` is in
    - `dataset`: dataset name without `-with-param`. e.g. `hand2-single-color`
- Run training specifying config file

    ```CUDA_VISIBLE_DEVICES=0 python train_part_wise_NeRF_supervised.py --config configs/supervised_part_wise_NeRF/your_config.yml --num_workers 1```
    - With default config, 20 GB GPU memory is necessary
    - Please reduce `bs` or `num_rays` to reduce memory requirements
- DDP training 
    
    ```python train_part_wise_NeRF_supervised_ddp.py --config configs/supervised_part_wise_NeRF/your_config.yml --gpus 4 --num_workers 1```
- visualization

    `visualize/part_based_nerf.ipynb`


## Visualize results

- Random sampling
    ```
    python NARF_random_image.py --config ../configs/unsupervised_dual_learning/THUman/20210204_generator_nerf_replace_dim160.yml
    ```
  The results are saved to the same directory as the snapshots


## License
TODO: Add license based on https://github.com/rosinality/stylegan2-pytorch


## BibTex
```
@inproceedings{2021narf
  author    = {Noguchi, Atsuhiro and Sun, Xiao and Lin, Stephen and Harada, Tatsuya},
  title     = {Neural Articulated Radiance Field},
  booktitle = {International Conference on Computer Vision},
  year      = {2021},
}
```