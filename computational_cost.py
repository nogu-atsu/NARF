# Calculate computational costs

import argparse

import torch

from NARF.models.net import NeRFGenerator, Generator, NeRFAutoEncoder
from NARF.models.model_utils import random_ray_sampler
from NARF.utils import yaml_config
from train import create_dataloader


def main(config):
    dataset, _ = create_dataloader(config.dataset)
    compute_cost(config, dataset)


def compute_cost(config, dataset):
    torch.backends.cudnn.benchmark = True

    size = config.dataset.image_size
    out_dir = config.out_root
    out_name = config.out
    cnn_based = config.generator_params.cnn_based
    num_points_per_ray = config.generator_params.nerf_params.Nc + config.generator_params.nerf_params.Nf

    dataset = dataset[0]
    num_bone = dataset.num_bone
    intrinsics = dataset.intrinsics

    if config.auto_encoder:
        if not cnn_based:
            gen = NeRFAutoEncoder(config.generator_params, size, intrinsics, num_bone, ch=32)
        else:
            gen = Generator(config.generator_params,
                            in_dim=dataset.num_bone + dataset.num_valid_keypoints + 3,
                            num_bone=dataset.num_bone, ch=32, size=size,
                            intrinsics=dataset.cp.intrinsics, rgb=True, auto_encoder=True)
    elif not cnn_based:
        gen = NeRFGenerator(config.generator_params, size, intrinsics, num_bone,
                            ray_sampler=random_ray_sampler)
    else:
        gen = Generator(config.generator_params,
                        in_dim=dataset.num_bone + dataset.num_valid_keypoints + 3,
                        num_bone=dataset.num_bone, ch=32, size=size,
                        intrinsics=dataset.cp.intrinsics, rgb=True)
    num_params = sum([p.numel() for p in gen.parameters()])

    with open(f"{out_dir}/result/{out_name}/num_params.txt", "w") as f:
        txt = f"#Params: {num_params}\n" + \
              f"#FLPOs: {gen.flops * num_points_per_ray}" + \
              f"#Memory: {gen.memory_cost * num_points_per_ray}"
        f.write(txt)
    print(out_name)
    print("#Params", num_params)
    print("#FLPOs", gen.flops * num_points_per_ray)
    print("#Memmory", gen.memory_cost * num_points_per_ray)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="NARF/configs/default.yml")
    parser.add_argument('--default_config', type=str, default="NARF/configs/default.yml")
    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config)

    main(config)
