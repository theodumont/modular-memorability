import os
import sys
import gc
import numpy as np
import torch
import json
try:                from icecream import ic
except ImportError: ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
try:                from urllib import urlretrieve
except ImportError: from urllib.request import urlretrieve
import seaborn as sns
from config import cfg
sns.set()

def load_array_from_file(path):
    """Load a local file into a numpy array."""
    extension = path.split(".")[-1]
    if extension == "npy":
        array = np.load(path)
    elif extension == "txt":
        array = np.loadtxt(path)
        if array.ndim == 0:  # only for meanOF, for which we only have the mean
            array = np.expand_dims(array, axis=0)
        array = np.expand_dims(array, axis=0)
    else:
        raise NotImplementedError
    return array

def save_array_to_file(array, path):
    """Save an array to a local file."""
    extension = path.split(".")[-1]
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if extension == "npy":
        np.save(path, array)
    elif extension == "txt":
        np.savetxt(path, array)
    else:
        raise NotImplementedError


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)



def flatten_time(input, delta_frames=5):

    batch_size, num_frames, x, y, channels = input.shape
    input = input.reshape(batch_size * num_frames // delta_frames, delta_frames, x, y, channels)[:,0,:,:,:]

    return input

def merge_by_video(output, batch_size):

    flat_batch_size, _ = output.shape
    output = output.reshape(batch_size, flat_batch_size // batch_size, 1).mean(axis=1)

    return output


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_seeds(seed):
    """Set seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU


from scipy import stats
import matplotlib.pyplot as plt

def draw_hist(scores_gt, scores_pr):
    min_x = min(scores_gt.min(), scores_pr.min())
    max_x = max(scores_gt.max(), scores_pr.max())

    hist_gt = np.histogram(scores_gt, bins=64, range=(min_x, max_x))
    hist_pr = np.histogram(scores_pr, bins=64, range=(min_x, max_x))

    fig = plt.figure(figsize=(7,5))
    plt.hist(scores_gt, density=True, bins=64, alpha=.5, label="ground truth")
    plt.hist(scores_pr, density=True, bins=64, alpha=.5, label="predictions")
    plt.xlabel("Memorability score")
    plt.ylabel("Distribution")
    plt.legend()

    return hist_gt, hist_pr, fig

def draw_dist_rank_1(scores_gt, scores_pr):
    scores_gt = scores_gt[:,0]
    scores_pr = scores_pr[:,0]
    ranks_gt = np.argsort(np.argsort(- scores_gt))
    ranks_pr = np.argsort(np.argsort(- scores_pr))
    fig = plt.figure(figsize=(7,5))
    plt.scatter(ranks_gt, scores_gt, label="ground truth", marker='+')
    plt.scatter(ranks_pr, scores_pr, label="predictions", marker='+')
    plt.xlabel("Image rank N")
    plt.ylabel("Memorability score")
    plt.legend()

    return fig

def draw_dist_rank_2(scores_gt, scores_pr):
    scores_gt = scores_gt[:,0]
    scores_pr = scores_pr[:,0]
    ranks_gt = np.argsort(np.argsort(- scores_gt))
    ranks_pr = np.argsort(np.argsort(- scores_pr))
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w
    scores_pr = moving_average(scores_pr, 20)
    fig = plt.figure(figsize=(7,5))
    plt.scatter(ranks_gt, scores_gt, label="ground truth", marker='+')
    plt.scatter(ranks_gt, scores_pr, label="predictions", marker='+')
    plt.xlabel("Image rank N")
    plt.ylabel("Memorability score")
    plt.legend()

    return fig

import pickle
def _get_data(dataset_name, splits, root_dir):
    data = []
    if dataset_name == "Memento10k":
        for split in splits:
            with open(os.path.join(root_dir, f"memento_{split}_data.json"), 'r') as _data:
                data += json.load(_data)
    elif dataset_name == "LaMem":
        for split in splits:
            data.append(np.array(np.loadtxt(os.path.join(root_dir, "splits", split + '_1.txt'), delimiter=' ', dtype=str)))
        data = np.concatenate(data, axis=0)
    elif dataset_name == "VideoMem":
        with open(os.path.join(root_dir, "train_test_split_videomem.pkl"), 'rb') as f:
            filenames_tuple = pickle.load(f)
        split_idxs = {"train": 0, "val": 1}
        for split in splits:
            filenames = filenames_tuple[split_idxs[split]]
            filenames = [filename.replace(".webm", ".mp4") for filename in filenames]
            data.append(np.array(filenames))
        data = np.concatenate(data, axis=0)
    else:
        raise NotImplementedError
    return data
def get_train_data(dataset_name):
    return _get_data(dataset_name, splits=["train"], root_dir=cfg.DIR.ROOT_DIRS[dataset_name])
def get_all_data(dataset_name):
    return _get_data(dataset_name, splits=["train", "val"], root_dir=cfg.DIR.ROOT_DIRS[dataset_name])