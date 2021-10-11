
# THUman dataset

Dataset preparation code is based on https://github.com/ZhengZerong/DeepHuman

## Requirements
- python 2.7
- numpy
- opendr
- opencv-python


## Dataset preparation
- Download SMPL model from https://smpl.is.tue.mpg.de/en and unzip here.
  - Rename `smpl/model/basicModel_f_lbs_10_207_0_v1.0.0.pkl` to `smpl/model/basicmodel_f_lbs_10_207_0_v1.0.0.pkl`
- Download THUman dataset from https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset and unzip it.
    - Specify the path to `dataset_dir` in config files `configs/*.py`
- update `data_list.txt` in `dataset` if necessary 
- Specify the path to save the data to `output_root_dir` in config files `configs/*.py`
- Run `python render_THUman.py --config_path configs/[dataset_name].py`

## Dataset name
- all
  - Training data for NARF autoencoder.
- results_gyx_20181017_lst_1_F
  - Female data used in the paper
- results_wxl_20181008_wlz_3_M
  - Male data used in the paper


## Acknowledgement
```
@InProceedings{Zheng2019DeepHuman, 
    author = {Zheng, Zerong and Yu, Tao and Wei, Yixuan and Dai, Qionghai and Liu, Yebin},
    title = {DeepHuman: 3D Human Reconstruction From a Single Image},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
@article{SMPL:2015,
      author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
      title = {{SMPL}: A Skinned Multi-Person Linear Model},
      journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }
```