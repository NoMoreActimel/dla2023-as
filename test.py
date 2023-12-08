import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_as.loss as module_loss
import hw_as.model as module_model
from hw_as.utils import ROOT_PATH
from hw_as.utils.object_loading import get_dataloaders
from hw_as.utils.parse_config import ConfigParser
from inference import run_inference

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = module_model.HiFiGANModel(config.config)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)

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
    
    run_inference(
        model=model,
        dataset=dataloaders["val"].dataset,
        indices=[30, 40, 50],
        dataset_type="val",
        inference_path="data/datasets/ljspeech/inference",
        compute_losses=True,
        output_loss_filepath="output.json",
        criterion=criterion,
        epoch=None
    )
    
    print(f"Generation results written into data/datasets/ljspeech/inference/val folder")
    print(f"Losses in output.json")

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
        default=5,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
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
            "val": {
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

    assert config.config.get("data", {}).get("val", None) is not None
    config["data"]["val"]["batch_size"] = args.batch_size
    config["data"]["val"]["n_jobs"] = args.jobs

    main(config)
