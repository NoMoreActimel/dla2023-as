import argparse
import json
import numpy as np
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_as.loss as module_loss
import hw_as.model as module_arch

from hw_as.metric.eer_metric import EERMetric
from hw_as.utils import ROOT_PATH
from hw_as.utils import prepare_device, get_number_of_parameters
from hw_as.utils.object_loading import get_dataloaders
from hw_as.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = module_arch.RawNet2Model(config.config["model"]["args"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print(f"Number of model parameters: {get_number_of_parameters(model)}")

    # get function handles of loss and metrics
    criterion = config.init_obj(config["loss"], module_loss).to(device)
    
    metric = EERMetric()

    len_val_epoch = config["trainer"].get("len_val_epoch", len(dataloaders["test"]))

    targets, predicts = [], []
    losses = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloaders["test"]), desc="test", total=len_val_epoch):
            for tensor_for_gpu in ["wav", "target"]:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
            batch["predict"] = model(**batch)
            losses.append(criterion(**batch).item())
            
            targets.extend(batch["target"].detach().cpu().tolist())
            predicts.extend(batch["predict"][:, 1].detach().cpu().tolist())

            if batch_idx >= len_val_epoch:
                break

    loss = np.mean(losses)    
    targets, predicts = np.array(targets), np.array(predicts)
    EER = metric(predict=predicts, target=targets)

    print(f"Cross-Entropy loss on test: {loss}")
    print(f"EER on test: {EER}")
    

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
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=None,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=None,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None, \
        f"No test dataset provided!"
    if args.batch_size is not None:
        config["data"]["test"]["batch_size"] = args.batch_size
    if args.jobs is not None:
        config["data"]["test"]["n_jobs"] = args.jobs

    main(config)
