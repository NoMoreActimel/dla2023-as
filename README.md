# Voice Anti-spoofing Homework

This is a repository with the as homework of the HSE DLA Course. It includes the implementation of FastSpeech2 model architecture and all training utilities. The training was performed on the LJSpeech dataset.

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

You need to download MFA alignments for LJSpeech and WaveGlow for spec-to-wav inference. All files are stored on google drive, first install gdown:
```shell
pip install gdown
```

You can download processed data, if you do not want to perform alignments or change something in data processing pipeline. First option is to download full-data archive, having data and raw_data in it (including initial wavs and texts).
```shell
gdown --id 1J-Fv9dNxFPMlzqQMTRL_pRmciUmAd7rE
mkdir -p {{ROOT_PATH}}/dla2023-as/data/datasets/ljspeech/
unzip data_processed.zip
mv content/dla2023-as/data/datasets/ljspeech/* data/datasets/ljspeech/
```
You can download data without initial wavs and texts, which takes much less space:
```shell
gdown --id 1--ZVSJlnzBvvSdC7g1b3oYKQo_IpGbgC
mkdir -p {{ROOT_PATH}}/dla2023-as/data/datasets/ljspeech/data/
unzip data_processed.zip
mv content/dla2023-as/data/datasets/ljspeech/data/* data/datasets/ljspeech/data/
```

If you do not want to download all processed data, then MFA alignments are available by themselves:
MFA alignments:
```shell
gdown --id 14mi82n1FxXZ0XHLKpO9DmpaFBQCkyJ-X
unzip {{ROOT_PATH}}/LJSpeech.zip
```

Finally, you will need WaveGlow for inference:
```shell
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p {{ROOT_PATH}}/dla2023-as/waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt {{ROOT_PATH}}/dla2023-as/waveglow/pretrained_model/waveglow_256channels.pt
```

You may now launch training / testing of the model, specifying the config file. The default model config is given as default_test_config.json. However, you may check for other examples in hw_as/configs/as directory.


Overall, to launch pretrained model you need to download the model-checkpoint and launch the test.py:
```shell
gdown --id 1X1B5qX4Ojeo4u569EmhoBZEQRxcU1t0O
gdown --id 12JkbM8smVxqiqzDIXow_ervyDqkfqv4T
mkdir default_test_model
mv checkpoint-epoch95.pth default_test_model/checkpoint.pth
mv config.json default_test_model/config.json
```
```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -o test_result.json
``` 


## Structure

All written code is located in the hw_as repository. Scripts launching training and testing are given as train.py and test.py in the root project directory. First one will call the trainer instance, which is the class used for training the model. For the convenience everything is getting logged using the wandb logger, you may also look audios and many interesting model-weights graphs out there.

## Training

To train the model you need to specify the config path:
```shell
python3 train.py -c hw_as/configs/config_name.json
```
If you want to proceed training process from the saved checkpoint, then:
```shell
python3 train.py -c hw_as/configs/config_name.json -r saved/checkpoint/path.pth
```

## Testing

Some basic tests are located in hw_as/tests directory. Script to run them:

```shell
python3 -m unittest discover hw_as/tests
```
