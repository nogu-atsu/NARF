# Distributed data parallel

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from NARF.utils import yaml_config, ddp_data_sampler
from train import create_dataset, train_func, validation_func, cache_dataset


def ddp_train(rank, world_size, backend, config, validation=False):
    print("Running DDP train on rank {} in {}.".format(rank, world_size))

    # initialize the process group
    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=world_size)
    torch.manual_seed(0)

    dataset, data_loader = create_dataloader(config.dataset, rank, world_size)
    if validation:
        validation_func(config, dataset, data_loader, rank, ddp=True)
    else:
        train_func(config, dataset, data_loader, rank, ddp=True, world_size=world_size)

    dist.destroy_process_group()


def create_dataloader(config_dataset, rank, world_size):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers
    dataset_train, datasets_val = create_dataset(config_dataset)
    sampler_train = ddp_data_sampler(dataset_train, rank, world_size, shuffle, drop_last)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler_train)

    val_loaders = {}
    for key, dataset_val in datasets_val.items():
        sampler_val = ddp_data_sampler(dataset_val, rank, world_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(dataset_val, batch_size=1, num_workers=num_workers, sampler=sampler_val)
        val_loaders[key] = val_loader
    return (dataset_train, datasets_val), (train_loader, val_loaders)


# for the master node
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="NARF/configs/default.yml")
    parser.add_argument('--default_config', type=str, default="NARF/configs/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--validation', action="store_true")
    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    cache_dataset(config.dataset)

    gpus = args.gpus
    nodes = args.nodes

    world_size = gpus * nodes
    backend = config.backend
    master_addr = config.master_addr
    master_port = config.master_port
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print("starting mp...")
    mp.spawn(ddp_train,
             args=(world_size, backend, config, args.validation),
             nprocs=world_size,
             join=True)
