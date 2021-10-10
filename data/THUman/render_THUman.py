# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function, absolute_import, division
import argparse
import imp
import copy
import os
import sys
from multiprocessing import Pool

import chumpy as ch
import cv2 as cv
import numpy as np

import renderers as rd
import util
from ObjIO import load_obj_data
from smpl.smpl_webuser.lbs import global_rigid_transformation
from smpl.smpl_webuser.serialization import ready_arguments

log = util.logger.write


def make_output_dir(out_dir):
    """creates output folders"""
    util.safe_mkdir(out_dir)
    util.safe_mkdir(os.path.join(out_dir, 'color'))
    util.safe_mkdir(os.path.join(out_dir, 'bone_params'))


def load_data_list(data_list_fname):
    """loads the list of 3D textured models"""
    # data_list_fname = os.path.join(dataset_dir, 'data_list.txt')
    data_list = []
    with open(data_list_fname, 'r') as fp:
        for line in fp.readlines():
            data_list.append(line[:-1])  # discard line ending symbol
    log('data list loaded. ')
    return data_list


def load_bg_list(bg_dir):
    """loads the list of background images"""
    bg_list_fname = os.path.join(bg_dir, 'img_list.txt')
    bg_list = []
    with open(bg_list_fname, 'r') as fp:
        for line in fp.readlines():
            bg_list.append(line[:-1])  # discard line ending symbol
    log('background list loaded. ')
    return bg_list


def check_rendered_img_existence(output_dir, model_idx):
    """check whether the model has been processed"""
    rendered = True
    img_idices = [4 * model_idx, 4 * model_idx + 1, 4 * model_idx + 2, 4 * model_idx + 3]
    for ind in img_idices:
        p = dict()
        p['color'] = '%s/color/color_%08d.jpg' % (output_dir, ind)
        p['normal'] = '%s/normal/normal_%08d.png' % (output_dir, ind)
        p['mask'] = '%s/mask/mask_%08d.png' % (output_dir, ind)
        p['vmap'] = '%s/vmap/vmap_%08d.png' % (output_dir, ind)
        p['params'] = '%s/params/params_%08d.json' % (output_dir, ind)

        for pi in p.values():
            rendered = rendered and os.path.exists(pi)
    return rendered


def load_models(dataset_dir, data_item):
    """loads the model and corrects the orientation"""
    mesh_dir = os.path.join(dataset_dir, data_item, 'mesh.obj')
    mesh = load_obj_data(mesh_dir)
    return mesh


def load_smpl_params(dataset_dir, data_item):
    smpl_params = os.path.join(dataset_dir, data_item, 'smpl_params.txt')

    with open(smpl_params, 'r') as fp:
        lines = fp.readlines()
        lines = [l[:-2] for l in lines]  # remove '\r\n'

        betas_data = filter(lambda s: len(s) != 0, lines[1].split(' '))
        betas = np.array([float(b) for b in betas_data])

        root_mat_data = lines[3].split(' ') + lines[4].split(' ') + \
                        lines[5].split(' ') + lines[6].split(' ')
        root_mat_data = filter(lambda s: len(s) != 0, root_mat_data)
        root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))
        root_rot = root_mat[:3, :3]
        root_trans = root_mat[:3, 3]

        theta_data = lines[8:80]
        theta = np.array([float(t) for t in theta_data])
    gender = data_item.split("/")[0][-1].lower()

    ## Load SMPL model
    ## Make sure path is correct

    fname = 'smpl/models/basicmodel_{0}_lbs_10_207_0_v1.0.0.pkl'.format(gender)

    dd = ready_arguments(fname)

    args = {
        'pose': dd['pose'],
        'J': dd['J'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
    }

    A, A_global = global_rigid_transformation(**args)

    for k, v in dd.items():
        setattr(A, k, v)

    # ## Apply shape & pose parameters
    A.pose[:] = theta
    A.betas[:] = betas
    bone_pose = np.stack([ag.r for ag in A_global])

    return bone_pose, root_rot, root_trans


def axis_transformation(mesh, bone_pose, axis_transformation):
    util.flip_axis_in_place(mesh, axis_transformation)
    bone_pose[:, :3] *= axis_transformation[None, :, None]
    return mesh, bone_pose


def transform_model_randomly(mesh, bone_pose):
    """translates the model to the origin, and rotates it randomly"""
    # random rotation
    y_rot = np.random.uniform(-np.pi, np.pi)
    x_rot = np.random.uniform(-0.3, 0.3)
    z_rot = np.random.uniform(-0.3, 0.3)
    # random scale
    scale = np.random.uniform(1.0, 1.5)
    # # mul(mat, mesh.v)
    bone_pose = util.rotate_pose_in_place(bone_pose, x_rot, y_rot, z_rot)
    mesh = util.rotate_model_in_place(mesh, x_rot, y_rot, z_rot)

    util.transform_mesh_in_place(mesh, scale=scale)
    util.transform_pose_in_place(bone_pose, scale=scale)

    # create a dict of transformation parameters
    param = dict()
    param['scale'] = scale
    param['rot'] = np.array([x_rot, y_rot, z_rot])
    return mesh, bone_pose, param


def move_to_origin(mesh, bone_pose, scale=0.5):
    """translates the model to the origin"""
    left_hip = 1
    right_hip = 2
    trans = -bone_pose[[left_hip, right_hip], :3, 3].mean(axis=0)
    util.transform_mesh_in_place(mesh, trans, scale)
    util.transform_pose_in_place(bone_pose, trans, scale)
    return mesh, bone_pose


# def draw_bone(img, keypoints):
#     for kp in keypoints:
#         cv.drawMarker(img, tuple(kp.astype("int")), (255, 0, 0, 255), markerType=cv.MARKER_SQUARE, markerSize=4)
#     return img


def save_rendered_data(img, pose_on_image, output_dir, img_idx, pose_to_camera, pose_to_world):
    """saves rendered images with correct format"""
    img = np.uint8(img * 255)
    cv.imwrite('%s/color/color_%08d.png' % (output_dir, img_idx),
               img[:, :, [2, 1, 0, 3]])

    # img_with_pose = draw_bone(img, pose_on_image)
    # cv.imwrite('%s/color/color_with_bone_%08d.png' % (output_dir, img_idx),
    #            img_with_pose[:, :, [2, 1, 0, 3]])

    np.save('%s/color/pose_to_camera_%08d.npy' % (output_dir, img_idx), pose_to_camera)
    np.save('%s/color/pose_to_world_%08d.npy' % (output_dir, img_idx), pose_to_world)


class Render:
    def __init__(self, dataset_type, random_lighting=True):
        self.render_setting = conf.render_setting[dataset_type]
        self.random_lighting = random_lighting
        if not random_lighting:
            self.vl_pos, self.vl_clr = util.sample_verticle_lighting(3, random=False)
            self.sh = util.sample_sh_component(random=False)

    def sample_view(self, novel_view):
        if novel_view:
            cam_t = ch.array((0, 0, np.random.uniform(1.5, 2.5)))

            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            cam_t = ch.array((0, 0, 2.0))

            theta = np.random.uniform(0, 0.3)
            phi = np.random.uniform(0, 2 * np.pi)
            angle = np.random.uniform(0, 2 * np.pi)
        cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
        cam_r = ch.array(cam_r * angle)
        return cam_t, cam_r

    def __call__(self, args):
        np.random.seed()
        di, data_item = args
        print("Processing data #", di)
        render_img_w = conf.render_img_w
        render_img_h = conf.render_img_h
        dataset_dir = conf.dataset_dir
        num_render_per_obj = self.render_setting["num_render_per_obj"]
        output_dir = self.render_setting["output_dir"]
        novel_view = self.render_setting["novel_view"]

        # preprocess 3D models
        mesh = load_models(dataset_dir, data_item)

        # align to smpl bone
        bone_pose, root_rot, root_trans = load_smpl_params(dataset_dir, data_item)
        mesh = util.translate_model_inplace(mesh, root_rot, root_trans)

        mesh, bone_pose = axis_transformation(mesh, bone_pose, conf.axis_transformation)

        # ---------- preprocess is done
        # save bone params
        np.save('%s/bone_params/bone_params_%08d.npy' % (output_dir, di), bone_pose)

        img_indices = np.arange(num_render_per_obj * di, num_render_per_obj * (di + 1))

        if not self.random_lighting:  # use one lighting condition
            vl_pos, vl_clr, sh = self.vl_pos, self.vl_clr, self.sh

        for vi in range(num_render_per_obj):
            mesh_ = copy.deepcopy(mesh)
            bone_pose_ = copy.deepcopy(bone_pose)
            mesh_, bone_pose_ = move_to_origin(mesh_, bone_pose_)
            mesh_, bone_pose_, param_0 = transform_model_randomly(mesh_, bone_pose_)
            img_ind = img_indices[vi]

            if self.random_lighting:  # random lighting condition
                vl_pos, vl_clr = util.sample_verticle_lighting(3)
                sh = util.sample_sh_component()

            cam_t, cam_r = self.sample_view(novel_view)
            img, pose_to_camera, pose_on_image = rd.render_training_pairs(mesh_, bone_pose_,
                                                                          render_img_w, render_img_h,
                                                                          cam_r, cam_t,  # bg,
                                                                          sh_comps=sh,
                                                                          light_c=ch.ones(3),
                                                                          vlight_pos=vl_pos,
                                                                          vlight_color=vl_clr)
            save_rendered_data(img, pose_on_image, output_dir, img_ind, pose_to_camera, bone_pose_)

            if hasattr(conf, "autoencoder") and conf.autoencoder:
                dir, base = os.path.split(output_dir)
                output_dir_view2 = os.path.join(dir + "_view2", base)
                cam_t, cam_r = self.sample_view(novel_view=False)
                img, pose_to_camera, pose_on_image = rd.render_training_pairs(mesh_, bone_pose_,
                                                                              render_img_w, render_img_h,
                                                                              cam_r, cam_t,  # bg,
                                                                              sh_comps=sh,
                                                                              light_c=ch.ones(3),
                                                                              vlight_pos=vl_pos,
                                                                              vlight_color=vl_clr)
                save_rendered_data(img, pose_on_image, output_dir_view2, img_ind, pose_to_camera, bone_pose_)


def render_data(dataset_type):
    output_dir = conf.render_setting[dataset_type]["output_dir"]
    data_list_fname = conf.data_list_fname
    random_lighting = conf.random_lighting

    np.random.seed()
    make_output_dir(output_dir)
    if hasattr(conf, "autoencoder") and conf.autoencoder:
        dir, base = os.path.split(output_dir)
        make_output_dir(os.path.join(dir + "_view2", base))

    data_list = load_data_list(data_list_fname)

    render = Render(dataset_type, random_lighting=random_lighting)
    # # if single process
    # for di, data_item in enumerate(data_list):
    #     render((di, data_item))

    # if multi process
    pool = Pool(20)
    try:
        pool.map_async(render, list(enumerate(data_list))).get(9999999)
    except KeyboardInterrupt:
        pool.terminate()
        print('KeyboardInterrupt!')
        sys.exit(1)


def main():
    for dataset_type in ["train", "same_view", "novel_view"]:
        render_data(dataset_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess THUman dataset')
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    conf = imp.load_source('module.name', args.config_path)
    main()
