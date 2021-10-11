
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
