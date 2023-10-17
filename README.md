#  Modular Memorability: Tiered Representations for Video Memorability Prediction

This repository contains an implementation of M3-S for the CVPR 2023 paper "Modular Memorability: Tiered Representations for Video Memorability Prediction", that achieves the following Spearman rank correlation scores:

|             | Memento10k | VideoMem |
| ----------- |:----------:|:--------:|
| M3-S (ours) |   0.670    |  0.563   |



## Code overview
Here are the main files and folders:
- `train.py`
- `config.py`: defines the training constants (*e.g.* *batch size*, *learning rate*, *nb of epochs*) as well as the path variables
- `model/`
- `dataset.py`



## Installation
1. **Requirements.** We provide a conda environment for the packages installation.
    ```bash
    # conda env
    conda env create -f envs/modular-mem.yml
    conda activate modular-mem

    # install HRNet
    pip install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
    # install torchsort
    pip install torchsort
    # download pretrained models for HRNet and set up CSN
    bash download_pretrained.sh
    # (only to generate features) extract pre-computed .pickle files for PCA
    tar -xvf pickle.tar; rm pickle.tar
    ```
2. **Datasets & features.**
    - **Datasets:** The [Memento10k](http://memento.csail.mit.edu/) and [VideoMem](https://www.interdigital.com/data_sets/video-memorability-dataset) datasets have to be downloaded beforehand. The directories should have the following structure:
        ```bash
        /Memento10k/
        ├── videos_npy/
        ├── memento_test_data.json
        ├── memento_train_data.json
        └── memento_val_data.json
        /VideoMem/
        ├── resized_mp4/
        ├── ground-truth_dev-set.csv
        └── train_test_split_videomem.pkl
        ```
        where `videos_npy` contains the videos of Memento10k in `.npy` format and `resized_mp4` contains the videos of VideoMem in `.mp4` format, resized to 256x256. By default, it is assumed that the datasets are in a `datasets` directory, located at the same level as the `modular-memorability` folder, but this behavior can be changed in the `config.py` file.
    - **Raw, semantic and similarity features:** To generate the features, use the `gen_features.py` script:
        ```bash
        python3 gen_features.py
        ```
        The features will be generated in both `datasets/${DATASET_NAME}` folders (where `${DATASET_NAME}` is either `Memento10k` or `VideoMem`).


## Evaluation

We provide some model weights below:

| Dataset    | Spearman RC | Weights                                        | File size |
| ---------- |:-----------:| ---------------------------------------------- | ---------:|
| Memento10k |   0.6355    | [m3s_memento10k](./weights/m3s_memento10k.tar) |     7.3MB |
| VideoMem   |   0.5158    | [m3s_videomem](./weights/m3s_videomem.tar)     |     7.0MB |

To reproduce these two specific training results, you can run:
```bash
python3 train.py --dataset Memento10k --seed 3          --batch_size 32 --lr 0.001                 --epochs 20 --loss mse             --scheduler_gamma 0.2                --scheduler_step_size 5 --weight_decay 1e-5                   --use_raw --raw_features hog brightness contrast meanOF blurriness size_orig --use_similarity --similarity_methods dbscan --similarity_methods hrnet ip_csn_152 --use_hrnet --use_csn --use_similarity --similarity_methods dbscan --similarity_methods hrnet ip_csn_152
python3 train.py --dataset VideoMem   --seed 1528126568 --batch_size 32 --lr 0.0004057400860408805 --epochs 20 --loss mse_to_spearman --scheduler_gamma 0.2852061798531863 --scheduler_step_size 8 --weight_decay 1.1226272823355355e-05 --use_raw --raw_features hog brightness contrast meanOF blurriness size_orig --use_similarity --similarity_methods dbscan --similarity_methods hrnet ip_csn_152 --use_hrnet --use_csn --use_similarity --similarity_methods dbscan --similarity_methods hrnet ip_csn_152
```
There are the exact same parameters we used, and most of them are already the default in `config.py`.


## Training

### Basics
In order to train a model, simply choose the default hyperparameters in the `config.py` file (you can also specify them in command line), then run:
```bash
# default hyperparameters
python3 train.py
# custom hyperparameters
python3 train.py --lr 0.0005 --batch_size 16
```

### Loading a checkpoint
_To do_

### Using wandb
In order to log the results using [wandb](https://wandb.ai), you have to fill the `__C.CONST.WANDB_X` variables in the `config.py` file, and specify the `--log` (`-l`) argument during training.


### Command-line arguments (details)
Running `python3 train.py --help` gives:
```
usage: train.py [-h] [-n NAME] [-s] [-l] [-i] [--seed SEED] [--group GROUP] [--tags TAGS [TAGS ...]] [--data_augmentation] [--use_raw] [--raw_features RAW_FEATURES [RAW_FEATURES ...]] [--normalize_raw] [--use_temporal_std] [--fixed_features] [--use_hrnet] [--use_csn] [--hrnet_frames {1,5,9}] [--csn_arch {ip_csn_152,ir_csn_152}] [--use_similarity] [--similarity_features SIMILARITY_FEATURES [SIMILARITY_FEATURES ...]] [--similarity_methods SIMILARITY_METHODS [SIMILARITY_METHODS ...]] [--normalize_similarity] [--hidden_channels HIDDEN_CHANNELS [HIDDEN_CHANNELS ...]] [--dataset {Memento10k,LaMem}] [--batch_size BATCH_SIZE] [--loss {mse,l1,spearman,mse_to_spearman}] [--lr LR] [--epochs EPOCHS] [--optimizer OPTIMIZER] [--weight_decay WEIGHT_DECAY] [--scheduler SCHEDULER] [--scheduler_gamma SCHEDULER_GAMMA] [--scheduler_step_size SCHEDULER_STEP_SIZE] [--checkpoint_local] [--checkpoint_run_id CHECKPOINT_RUN_ID] [--checkpoint_epoch CHECKPOINT_EPOCH]

Training script for memorability model

optional arguments:
  -h, --help                                          show this help message and exit
  -n NAME, --name NAME                                name to append to run id
  -s, --save                                          whether saving weights
  -l, --log                                           whether logging in WANDB
  -i, --log_images                                    whether logging images in WANDB
  --seed SEED                                         random seed, will be set for numpy and pytorch
  --group GROUP                                       to group runs in wandb
  --tags TAGS                                         tags in wandb
  --data_augmentation                                 whether using data augmentation for input video, requires fixed_features set to False
  --use_raw                                           whether using the raw module
  --raw_features RAW_FEATURES                         list of raw features
  --normalize_raw                                     whether normalizing them
  --use_temporal_std                                  whether using the temporal std of the raw features
  --fixed_features                                    whether using pre-computed features
  --use_hrnet                                         whether using the HRNet module
  --use_csn                                           whether using the CSN module
  --hrnet_frames {1,5,9}                              nb frames as input to HRNet
  --csn_arch {ip_csn_152,ir_csn_152}                  CSN architecture
  --use_similarity                                    whether using the similarity module
  --similarity_features SIMILARITY_FEATURES           list of similarity features
  --similarity_methods SIMILARITY_METHODS             list of similarity methods
  --normalize_similarity                              whether normalizing them
  --hidden_channels HIDDEN_CHANNELS                   MLP hidden layers
  --dataset {Memento10k,LaMem,VideoMem}               dataset on which to train
  --batch_size BATCH_SIZE                             batch size
  --loss {mse,l1,mse_tails,spearman,mse_to_spearman}  loss function
  --lr LR                                             start learning rate
  --epochs EPOCHS                                     nb of epochs
  --optimizer OPTIMIZER                               optimizer name
  --weight_decay WEIGHT_DECAY                         optimizer weight decay (regularization)
  --scheduler SCHEDULER                               scheduler name
  --scheduler_gamma SCHEDULER_GAMMA                   coeff for learning rate decay
  --scheduler_step_size SCHEDULER_STEP_SIZE           step for lerning rate decay
  --checkpoint_local                                  using local storage or wandb cloud for loading checkpoint
  --checkpoint_run_id CHECKPOINT_RUN_ID               ID of checkpoint run in wandb
  --checkpoint_epoch CHECKPOINT_EPOCH                 epoch of checkpoint run in wandb
```


### Compression features with VideoMem
In order to use the compression features with VideoMem, one needs to uncomment the following line in `config.py`
```python
"dbscan": {"file_extension": ".txt", "folder_name": "dbscan",  "nb_features": 699 / 2},  # VideoMem
```
and comment this one
```python
"dbscan": {"file_extension": ".txt", "folder_name": "dbscan",  "nb_features": 852 / 2},  # Memento10k
```
