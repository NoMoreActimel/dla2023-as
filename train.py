import argparse
import collections
import warnings

import numpy as np
import torch

from itertools import chain

import hw_as.loss as module_loss
import hw_as.metric as module_metric
import hw_as.model as module_arch
from hw_as.trainer import Trainer
from hw_as.utils import prepare_device, get_number_of_parameters
from hw_as.utils.object_loading import get_dataloaders
from hw_as.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = module_arch.RawNet2Model(config.config["model"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print(f"Number of model parameters: {get_number_of_parameters(model)}")

    # get function handles of loss and metrics
    criterion = config.init_obj(
        config["loss"],
        module_loss
    ).to(device)
    
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=None)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler

    optimizer = config.init_obj(
        config["optimizer"],
        torch.optim,
        model.parameters()
    )
    lr_scheduler = config.init_obj(
        config["lr_scheduler"],
        torch.optim.lr_scheduler,
        optimizer
    )

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
