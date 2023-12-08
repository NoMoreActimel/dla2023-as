# Neural Vocoder Homework

This is a repository with the NV homework of the HSE DLA Course. It includes the implementation of HiFiGAN model architecture and all training utilities. The training was performed on the LJSpeech dataset.

## Installation guide

Clone the repository:
```shell
%cd {{local_path}}
git clone https://github.com/NoMoreActimel/dla2023-nv.git
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

Download preprocessed data - cutted wavs to the length of 16384 and generated mel-spectrograms on them. You can preprocess data by yourself using preprocessing flags "load_dataset", "preprocess_wavs" and "generate_mels" in confing.
```shell
!gdown https://drive.google.com/u/0/uc?id=1--qRbIsg_yLdW_4lvBOYoQea8hxWsuJz
!unzip -q {{local_path}}/dla2023-nv/nv-data-cut-nonorm.zip
!mkdir -p {{local_path}}/dla2023-nv/data/datasets/ljspeech/raw_data
!mkdir -p  {{local_path}}/dla2023-nv/data/datasets/ljspeech/data
!mv content/dla2023-nv/data/datasets/ljspeech/data/* {{local_path}}/dla2023-nv/data/datasets/ljspeech/data/
```

You may now launch training / testing of the model, specifying the config file. The default model config is given as hw_nv/configs/nv/hifigan_config.json.

Overall, to launch pretrained model you need to download the model-checkpoint and launch the test.py:
```shell
!gdown 1c6cx21UOjCbYwr3qRPptzME3tnhVFt3d
!gdown 1WF9-yJX5XR-Yo0oIPpuME6tl-pXjefTG
mkdir default_test_model
mv checkpoint-epoch320.pth default_test_model/checkpoint.pth
mv config.json default_test_model/config.json
```

```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth
``` 

## Structure

All written code is located in the hw_nv repository. Scripts launching training and testing are given as train.py and test.py in the root project directory. First one will call the trainer instance, which is the class used for training the model. For the convenience everything is getting logged using the wandb logger, you may also look audios and many interesting model-weights graphs out there.

## Training

To train the model you need to specify the config path:
```shell
python3 train.py -c hw_nv/configs/config_name.json
```
If you want to proceed training process from the saved checkpoint, then:
```shell
python3 train.py -c hw_nv/configs/config_name.json -r saved/checkpoint/path.pth
```

## Testing

Some basic tests are located in hw_nv/tests directory. Script to run them:

```shell
python3 -m unittest discover hw_nv/tests
```
