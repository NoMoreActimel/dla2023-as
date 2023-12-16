# Voice Anti-spoofing Homework

This is a repository with the as homework of the HSE DLA Course. It includes the implementation of RawNet2 model and all training utilities. The training was performed on the ASVspoof2019 dataset, with all audios being randomly cropped to 64000 frames - 4 seconds.

## Installation guide

We will use ASVspoof2019 kaggle dataset: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset. To install it, first of all download kaggle package with pip and load your kaggle.json file:
```shell
pip install -q kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Download dataset to data_dir provided in config file:
```shell
cd path/to/data/dir
kaggle datasets download -d awsaf49/asvpoof-2019-dataset
unzip avspoof-2019-dataset
```

Clone the repository:
```shell
%cd local/cloned/project/path
git clone https://github.com/NoMoreActimel/dla2023-as.git
```

Install and create pyenv:
```shell
pyenv install 3.9.7
cd path/to/local/cloned/project
~/.pyenv/versions/3.9.7/bin/python -m venv asr_venv
```

Install required packages:

```shell
pip install -r ./requirements.txt
```

You may now launch training / testing of the model, specifying the config file. The default model config is given as default_test_config.json. However, you may check for other examples in hw_as/configs/asv directory.

To install pretrained checkpoint, you'll need gdown:
```shell
pip install gdown
gdown --id 1sZ5c3tKO2rHOlIThZtFhnsSSPyAr4lBI
gdown --id 1y33E-QqThpnQOxKVfRbjwuusapidzjKf
mkdir model_checkpoint
mv checkpoint-epoch140.pth model_checkpoint/checkpoint.pth
mv config.json model_checkpoint/config.json
```

Now you can launch test.py. By default it measures CELoss and EER on test, limiting to len_val_epoch, specified in config.trainer. You can remove len_val_epoch, in order to test on whole test dataset.
```shell
python test.py -c model_checkpoint/config.json -r model_checkpoint/checkpoint.pth
``` 

The provided checkpoint hits EER of 0.0467 on the whole test dataset.

## Structure

All written code is located in the hw_as repository. Scripts launching training and testing are given as train.py and test.py in the root project directory. First one will call the trainer instance, which is the class used for training the model. For the convenience everything is getting logged using the wandb logger, you may also look audios and many interesting model-weights graphs out there.

## Training

To train the model you need to specify the config path:
```shell
python3 train.py -c hw_as/configs/asv/config_name.json
```
If you want to proceed training process from the saved checkpoint, then:
```shell
python3 train.py -c hw_as/configs/asv/config_name.json -r saved/checkpoint/path.pth
```

## Testing

To launch testing you'll need ckeckpoint having its own config in the same /saved/checkpoint/ directory. You can specify config.json by yourself, though in order to load checkpoint model architectures must be the same.
```shell
python test.py -c hw_as/configs/asv/config_name.json -r saved/checkpoint/path.pth
``` 
