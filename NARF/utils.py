import argparse
import os
import subprocess
import sys

import torch
import yaml
from easydict import EasyDict as edict


def record_setting(out):
    """Record scripts and commandline arguments"""
    # out = out.split()[0].strip()
    source = out + "/source"
    if not os.path.exists(source):
        os.system('mkdir -p %s' % source)
        # os.mkdir(out)

    # subprocess.call("cp *.py %s" % source, shell=True)
    # subprocess.call("cp configs/*.yml %s" % out, shell=True)

    subprocess.call("find . -type d -name result -prune -o -name '*.py' -print0"
                    "| xargs -0 cp --parents -p -t %s" % source, shell=True)
    subprocess.call("find . -type d -name result -prune -o -name '*.yml' -print0|"
                    " xargs -0 cp --parents -p -t %s" % source, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def yaml_config(config, default_cofig, resume_latest=False, num_workers=1):
    default = edict(yaml.load(open(default_cofig), Loader=yaml.SafeLoader))
    conf = edict(yaml.load(open(config), Loader=yaml.SafeLoader))

    def copy(conf, default):
        for key in conf:
            if isinstance(default[key], edict):
                copy(conf[key], default[key])
            else:
                default[key] = conf[key]

    copy(conf, default)

    default.resume_latest = resume_latest
    default.dataset.num_workers = num_workers
    return default


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/hinge.yml")
    args = parser.parse_args()

    config = yaml_config(args.config)
    return config


def write(iter, loss, name, writer):
    writer.add_scalar("metrics/" + name, loss, iter)
    return loss


def ddp_data_sampler(dataset, rank, world_size, shuffle, drop_last):
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dist_sampler
