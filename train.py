import argparse
import collections
import warnings

import numpy as np
import torch

from itertools import chain

import hw_nv.loss as module_loss
import hw_nv.metric as module_metric
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser

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
    model = module_arch.HiFiGANModel(config.config)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    for module_name, n_params in model.get_number_of_parameters().items():
        print(f"Number of {module_name} parameters: {n_params}")

    # get function handles of loss and metrics
    criterion = {
        "generator": config.init_obj(
            config["loss"]["generator"],
            module_loss,
            mel_generator=dataloaders["train"].dataset.melspec_generator
        ).to(device),
        "discriminator": config.init_obj(
            config["loss"]["discriminator"],
            module_loss
        ).to(device)
    }
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=None)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    generator_params = model.generator.parameters()
    discriminator_params = chain(model.MPD.parameters(), model.MSD.parameters())

    optimizer = {
        "generator": config.init_obj(
            config["optimizer"]["generator"],
            torch.optim,
            generator_params
        ),
        "discriminator": config.init_obj(
            config["optimizer"]["discriminator"],
            torch.optim,
            discriminator_params
        )
    }

    lr_scheduler = {
        "generator": config.init_obj(
            config["lr_scheduler"]["generator"],
            torch.optim.lr_scheduler,
            optimizer["generator"]
        ),
        "discriminator": config.init_obj(
            config["lr_scheduler"]["discriminator"],
            torch.optim.lr_scheduler,
            optimizer["discriminator"]
        )
    }

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
