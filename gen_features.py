import os
import pickle
from tqdm import tqdm
from config import cfg
from utils import ic, save_array_to_file, load_array_from_file, get_train_data, get_all_data
from config import FEATURES_INFO
from sklearn.decomposition import PCA
# raw
import ffmpeg
import numpy as np
import imutils
import cv2
from PIL import Image, ImageStat
from skimage.feature import hog
# semantic
import torch
import torchvision.transforms as transforms
from transforms.video_transforms import ToTensor, NormalizeVideo
from dataset import DatasetMemento, DatasetLaMem, DatasetVideoMem
from model.m3s import get_hrnet, get_csn
# similarity
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import skvideo.io

# RAW FEATURES ===========================================================================

def _make_raw(feature_name, feature_fun, dataset_name):
    print(f"\n[RAW] Making {dataset_name} features for {feature_name}")

    root_dir = cfg.DIR.ROOT_DIRS[dataset_name]

    if feature_name == "hog":
        pca_name = f"pca_hog.pickle"
        pca_path = os.path.join(cfg.DIR.PICKLE, dataset_name, pca_name)
        if not os.path.isfile(pca_path):
            _make_pca_hog(dataset_name, root_dir, pca_path)
            # raise FileNotFoundError("Please create the pca.pickle file you specified.")
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)

    all_filenames = get_all_data(dataset_name)
    loop = tqdm(all_filenames, desc=f"Generating {feature_name} features for {dataset_name}")

    for i, record in enumerate(loop):
        if dataset_name == "Memento10k":
            # generate -------------------------------------------------------------------
            name       = record["filename"]
            input_path = os.path.join(root_dir, "videos_npy", name.replace(".mp4", ".npy"))
            input      = np.load(input_path)
            feature    = np.array([feature_fun(frame) for frame in input])
            if feature_name == "hog":
                feature = pca.transform(feature)
            res        = np.stack([feature.mean(axis=0), feature.std(axis=0)], axis=1)

            # save -----------------------------------------------------------------------
            feature_info = FEATURES_INFO[feature_name]
            feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".mp4", feature_info["file_extension"]))
        elif dataset_name == "LaMem":
            # generate -------------------------------------------------------------------
            name       = record[0]
            input_path = os.path.join(root_dir, "images", name)
            input      = np.array(Image.open(input_path).convert('RGB').resize((256, 256)))  # cubic interpolation
            feature    = feature_fun(input)
            if feature_name == "hog":
                feature = pca.transform(feature.reshape(1, -1)).squeeze(0)
            res        = np.stack([feature, np.zeros_like(feature)], axis=1)

            # save -----------------------------------------------------------------------
            feature_info = FEATURES_INFO[feature_name]
            feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".jpg", feature_info["file_extension"]))
        elif dataset_name == "VideoMem":
            # generate -------------------------------------------------------------------
            name       = record
            input_path = os.path.join(root_dir, "resized_mp4", name)
            input      = skvideo.io.vread(input_path)
            feature    = np.array([feature_fun(np.array(Image.fromarray(frame).resize((256,256))).astype(np.float32) / 255.) for frame in input])
            if feature_name == "hog":
                feature = pca.transform(feature)
            res        = np.stack([feature.mean(axis=0), feature.std(axis=0)], axis=1)

            # save -----------------------------------------------------------------------
            feature_info = FEATURES_INFO[feature_name]
            feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".mp4", feature_info["file_extension"]))
        else:
            raise NotImplementedError

        save_array_to_file(res, feature_path)

def _make_pca_hog(dataset_name, root_dir, pca_path):
    train_filenames = get_train_data(dataset_name)
    loop = tqdm(train_filenames, desc=f"Generating HOG train features for {dataset_name} (for PCA)")

    train_features = []
    for i, record in enumerate(loop):
        if dataset_name == "Memento10k":
            name       = record["filename"]
            input_path = os.path.join(root_dir, "videos_npy", name.replace(".mp4", ".npy"))
            input      = np.load(input_path)
            feature    = _get_hog(input[0])  # train PCA with first frame
            train_features.append(feature)
        elif dataset_name == "LaMem":
            name       = record[0]
            input_path = os.path.join(root_dir, "images", name)
            input      = np.array(Image.open(input_path).convert('RGB').resize((256, 256)))  # cubic interpolation
            feature    = _get_hog(input)
            train_features.append(feature)
        elif dataset_name == "VideoMem":
            name       = record
            input_path = os.path.join(root_dir, "resized_mp4", name)
            input = skvideo.io.vread(input_path)
            input = np.array(Image.fromarray(input[0]).resize((256,256))).astype(np.float32) / 255.

            feature    = _get_hog(input)  # train PCA with first frame
            train_features.append(feature)
        else:
            raise NotImplementedError

    pca = PCA(n_components=10, whiten=True)
    pca.fit(train_features)

    with open(pca_path, 'wb') as file:
        pickle.dump(pca, file)

def _get_hog(img):
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True, feature_vector=True)
    return fd
def _get_contrast(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = Image.fromarray(img)
    stats = ImageStat.Stat(img)
    return np.expand_dims(stats.stddev[0], axis=0)
def _get_contrast2(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = Image.fromarray(img)
    stats = ImageStat.Stat(img)
    return np.expand_dims(stats.stddev[2], axis=0)
def _get_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    brightness = hsv[0].mean()
    return np.expand_dims(brightness, axis=0)
def _get_blurriness(img, size=60):
    """
    Reference
    ---------
    https://www.pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
    """
    img = imutils.resize(img, width=500)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # grab the dimensions of the image and use the dimensions to derive the center
    # (x, y)-coordinates
    (h, w) = img.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift the zero frequency
    # component (i.e., DC component located at the top-left corner) to the center where it
    # will be more easy to analyze
    fft = np.fft.fft2(img)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low frequencies), apply the
    # inverse shift such that the DC component once again becomes the top-left, and then
    # apply the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image, then compute the mean of
    # the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    blur = np.mean(magnitude)

    return np.expand_dims(blur, axis=0)

def make_hog(dataset_name):
    _make_raw("hog", _get_hog, dataset_name)
def make_contrast(dataset_name):
    _make_raw("contrast", _get_contrast, dataset_name)
def make_contrast2(dataset_name):
    _make_raw("contrast", _get_contrast2, dataset_name)
def make_brightness(dataset_name):
    _make_raw("brightness", _get_brightness, dataset_name)
def make_blurriness(dataset_name):
    _make_raw("blurriness", _get_blurriness, dataset_name)
def make_size_orig(dataset_name):
    feature_name = "size_orig"
    print(f"\n[RAW] Making {dataset_name} features for {feature_name}")

    root_dir = cfg.DIR.ROOT_DIRS[dataset_name]
    all_filenames = get_all_data(dataset_name)
    loop = tqdm(all_filenames, desc=f"Generating {feature_name} features for {dataset_name}")

    if dataset_name == "Memento10k":
        TMP_DIR = "./tmp"
        os.makedirs(TMP_DIR, exist_ok=True)
        PREFIX_FINAL = "final"
        for record in loop:
            # GENERATE
            name         = record["filename"]
            filepath     = os.path.join(root_dir, "videos", name)
            file_final   = os.path.join(TMP_DIR, f'{PREFIX_FINAL}_{name}')
            # 1. removing audio from video ---------------------------------------------------
            stream       = ffmpeg.input(filepath).video
            stream       = ffmpeg.output(stream,   file_final,   vcodec="copy").global_args('-loglevel', 'quiet')
            ffmpeg.run(stream)
            # 2. getting size ----------------------------------------------------------------
            size_final   = os.path.getsize(file_final)
            # 3. saving ----------------------------------------------------------------------
            feature_info = FEATURES_INFO[feature_name]
            feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".mp4", feature_info["file_extension"]))
            save_array_to_file(np.expand_dims(size_final, axis=0), feature_path)
            # 4. cleaning --------------------------------------------------------------------
            os.remove(file_final)
        os.rmdir(TMP_DIR)

    elif dataset_name == "LaMem":
        for record in loop:
            name = record[0]
            file_final = os.path.join(root_dir, "images", name)
            # 2. getting size ----------------------------------------------------------------
            size_final = os.path.getsize(file_final)
            # 3. saving ----------------------------------------------------------------------
            feature_info = FEATURES_INFO[feature_name]
            feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".jpg", feature_info["file_extension"]))
            save_array_to_file(np.expand_dims(size_final, axis=0), feature_path)

    elif dataset_name == "VideoMem":
        TMP_DIR = "./tmp"
        os.makedirs(TMP_DIR, exist_ok=True)
        PREFIX_FINAL = "final"
        for record in loop:
            # GENERATE
            name         = record
            filepath     = os.path.join(root_dir, "resized_mp4", name)
            file_final   = os.path.join(TMP_DIR, f'{PREFIX_FINAL}_{name}')
            # 1. removing audio from video ---------------------------------------------------
            stream       = ffmpeg.input(filepath).video
            stream       = ffmpeg.output(stream,   file_final,   vcodec="copy").global_args('-loglevel', 'quiet')
            ffmpeg.run(stream)
            # 2. getting size ----------------------------------------------------------------
            size_final   = os.path.getsize(file_final)
            # 3. saving ----------------------------------------------------------------------
            feature_info = FEATURES_INFO[feature_name]
            feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".mp4", feature_info["file_extension"]))
            save_array_to_file(np.expand_dims(size_final, axis=0), feature_path)
            # 4. cleaning --------------------------------------------------------------------
            os.remove(file_final)
        os.rmdir(TMP_DIR)
def make_meanOF(dataset_name):
    feature_name = "meanOF"

    print(f"\n[RAW] Making {dataset_name} features for {feature_name}")

    assert dataset_name == "LaMem"
    root_dir = cfg.DIR.ROOT_DIRS[dataset_name]
    all_filenames = get_all_data(dataset_name)
    loop = tqdm(all_filenames, desc=f"Generating {feature_name} features for {dataset_name}")
    for record in loop:
        name = record[0]
        feature_info = FEATURES_INFO[feature_name]
        feature_path = os.path.join(root_dir, feature_info["folder_name"], name.replace(".jpg", feature_info["file_extension"]))
        save_array_to_file(np.expand_dims(0, axis=0), feature_path)


# SEMANTIC FEATURES ======================================================================

def _make_semantic(feature_name, dataset_name, hrnet_frames=None):
    print(f"\n[SEMANTIC] Making {dataset_name} features for {feature_name} ")

    root_dir = cfg.DIR.ROOT_DIRS[dataset_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(device)

    # LOADING AND FREEZING MODEL =========================================================
    if   feature_name == "hrnet":      model = get_hrnet()
    elif feature_name == "ip_csn_152": model = get_csn("ip_csn_152")
    else: raise NotImplementedError

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    # NORMALIZATION AND DATASET ==========================================================
    normalize_dict = {
        "ip_csn_152": {"mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]},
        "hrnet"     : {"mean": [0.485  , 0.456   , 0.406  ], "std": [0.229  , 0.224  , 0.225   ]},
    }


    if dataset_name == "Memento10k":
        transform = transforms.Compose([
            ToTensor(),
            NormalizeVideo(**normalize_dict[feature_name]),
        ])
        split = "train_val"
        dataset_class = DatasetMemento
    elif dataset_name == "LaMem":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_dict[feature_name]["mean"], normalize_dict[feature_name]["std"]),
        ])
        split = "all"
        dataset_class = DatasetLaMem
    elif dataset_name == "VideoMem":
        transform = transforms.Compose([
            ToTensor(),
            NormalizeVideo(**normalize_dict[feature_name]),
        ])
        split = "train_val"
        dataset_class = DatasetVideoMem

    dataset = dataset_class(
        split=split,
        data_augmentation=False,
        use_raw=False,
        raw_features=[],
        normalize_raw=False,
        use_temporal_std=False,
        fixed_features=False,
        use_hrnet=True,
        use_csn=True,
        hrnet_frames=1,
        csn_arch="ip_csn_152",
        use_similarity=False,
        similarity_methods=[],
        similarity_features=[],
        normalize_similarity=False,
        input_transform=transform,
        compute_raw=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=cfg.CONST.NUM_WORKERS)


    # GENERATING FEATURES ================================================================
    loop = tqdm(dataloader, desc=f"Generating {split} {feature_name} features for {dataset.__class__.__name__}")
    for i, sample in enumerate(loop):
        input = sample['input'].float().to(device)          # (1, 45, 256, 256, 3)

        # PASS THROUGH MODEL -------------------------------------------------------------
        if feature_name == "hrnet":
            if dataset_name == "Memento10k":
                input = input.permute(0,1,4,2,3)                # (1, 45, 3, 256, 256)
                input_hrnet = input[:,::input.shape[1] // hrnet_frames,:,:,:]
                features = [model(input_hrnet[:,i,:,:,:]).squeeze(-1).squeeze(-1) for i in range(input_hrnet.shape[1])]
                features = torch.stack(features).mean(axis=0).squeeze().detach().cpu()
            elif dataset_name == "LaMem":
                # input                                         # (1, 3, 256, 256)
                features = model(input).squeeze(-1).squeeze(-1).detach().cpu()
            elif dataset_name == "VideoMem":
                input = input.permute(0,1,4,2,3)                # (1, 45, 3, 256, 256)
                input_hrnet = input[:,::input.shape[1] // hrnet_frames,:,:,:]
                features = [model(input_hrnet[:,i,:,:,:]).squeeze(-1).squeeze(-1) for i in range(input_hrnet.shape[1])]
                features = torch.stack(features).mean(axis=0).squeeze().detach().cpu()
            else:
                raise NotImplementedError
        elif feature_name in ["ip_csn_152", "ir_csn_152"]:
            if dataset_name == "Memento10k":
                input = input.permute(0,4,1,2,3)                # (1, 3, 45, 256, 256)
                output = model(input)
                features = output.squeeze().detach().cpu()
            elif dataset_name == "LaMem":
                input = input.unsqueeze(2)
                # input = input.repeat(1,1,10,1,1)
                output = model(input)
                features = output.squeeze().detach().cpu()
            elif dataset_name == "VideoMem":
                input = input.permute(0,4,1,2,3)                # (1, 3, 45, 256, 256)
                output = model(input)
                features = output.squeeze().detach().cpu()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # SAVING FEATURES ----------------------------------------------------------------
        suffix = f"_{hrnet_frames}frames_avg" if feature_name == "hrnet" and dataset_name in ["Memento10k", "VideoMem"] else ""
        for feature, name in zip(features, sample['name']):
            if dataset_name == "Memento10k": origin_extension = ".mp4"
            elif dataset_name == "LaMem":    origin_extension = ".jpg"
            elif dataset_name == "VideoMem": origin_extension = ".mp4"
            save_array_to_file(
                feature,
                os.path.join(root_dir, feature_name, name.replace(origin_extension, suffix + ".npy"))
            )

def make_hrnet(dataset_name, hrnet_frames=1):
    _make_semantic("hrnet", dataset_name, hrnet_frames=hrnet_frames)
def make_csn(dataset_name):
    _make_semantic("ip_csn_152", dataset_name)


# SIMILARITY FEATURES ===================================================================

def _get_trained_pca(train_features, pca_comp):
    pca = PCA(n_components=pca_comp, whiten=True)
    pca.fit(train_features)
    return pca
def _get_trained_kde(train_features):
    bandwidths = 100 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(
        KernelDensity(kernel='epanechnikov'),
        {'bandwidth': bandwidths},
    )
    grid.fit(train_features[:10000])
    # scores = grid.cv_results_['mean_test_score']
    best_bandwidth = grid.best_params_['bandwidth']
    print(f"Best accuracy: {grid.best_score_:.3f} for bandwidth {best_bandwidth:.4f}")
    return grid.best_estimator_
def _get_or_make_pickle_object(type, train_features, feature_name, dataset_name, pickle_suffix="", verbose=True, save=True, **kwargs):
    pickle_path = os.path.join(cfg.DIR.PICKLE, dataset_name, f"{type}_{feature_name}{pickle_suffix}.pickle")

    if os.path.isfile(pickle_path) and not cfg.SIMILARITY.OVERWRITE_PICKLE:
        if verbose: print(f"Found {type.upper()} pickle file: {pickle_path}")
        with open(pickle_path, 'rb') as file:
            operator = pickle.load(file)
    else:
        if verbose: print(f"Cannot found {type.upper()} pickle file ({pickle_path}). Generating one...")
        if   type == "pca": operator = _get_trained_pca(train_features, kwargs['pca_comp'])
        elif type == "kde": operator = _get_trained_kde(train_features)
        if save:
            with open(pickle_path, 'wb') as file:
                pickle.dump(operator, file)
                print(f"Saved {pickle_path}")

    return operator

def _get_kde(train_features, all_features, feature_name, dataset_name, pickle_suffix="", verbose=True, save=True,):
    kde = _get_or_make_pickle_object("kde", train_features, feature_name=feature_name, dataset_name=dataset_name, pickle_suffix=pickle_suffix, verbose=verbose, save=save)
    logprobs = kde.score_samples(all_features)
    scores = - logprobs
    return scores
def _get_cosine(train_features, all_features, submethod, threshold=None):
    if submethod == "mean":
        sim = cosine_similarity(all_features, train_features.mean(axis=0, keepdims=True))
        scores = sim[:,0]
        return scores
    elif submethod == "threshold":
        sim = cosine_similarity(all_features)
        scores = np.count_nonzero(sim >= threshold, axis=0) / sim.shape[0]
        return scores
    else:
        raise NotImplementedError
def _get_fractional(train_features, all_features, submethod):
    f = cfg.SIMILARITY.FRACTIONAL.F
    if submethod == "mean":
        sim = cdist(all_features, train_features.mean(axis=0, keepdims=True), lambda u, v: np.sum(np.abs(u - v)**f)**(1 / f))
        scores = sim[:,0]
        return scores
    else:
        raise NotImplementedError
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
def _get_dbscan(train_features, all_features, feature_name):

    tsne = TSNE(n_components=3)
    train_tsne = tsne.fit_transform(train_features)
    # all_tsne = tsne.fit_transform(all_features)


    if feature_name == "hrnet": eps = cfg.SIMILARITY.DBSCAN.HRNET_EPS
    if feature_name == "ip_csn_152": eps = cfg.SIMILARITY.DBSCAN.CSN_EPS
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(train_tsne)
    train_predictions = dbscan.labels_
    nb_classes = train_predictions.max()
    class_ids = range(nb_classes)
    print(nb_classes, "classes for DBSCAN")

    # plt.figure(figsize=(20,20))
    # plt.scatter(x=train_tsne[:,0], y=train_tsne[:,1], marker='+', c=train_predictions, cmap='hsv', label=train_predictions)
    # plt.legend()
    # plt.savefig("./figs/similarity/dbscan_tsne_train")

    clf = MLPClassifier(max_iter=500).fit(train_features, train_predictions)
    all_classif_predictions = clf.predict(all_features)
    all_classif_predictions[:len(train_predictions)] = train_predictions
    nb_classes_classif = all_classif_predictions.max()
    print(nb_classes_classif, "classes for classif")

    # plt.figure(figsize=(20,20))
    # plt.scatter(x=all_tsne[:,0], y=all_tsne[:,1], marker='+', c=all_classif_predictions, cmap='hsv', label=train_predictions)
    # plt.legend()
    # plt.savefig("./figs/similarity/dbscan_tsne_pred")

    one_hot = np.eye(nb_classes_classif + 1)[all_classif_predictions]

    return one_hot

import itertools
def _get_distance_to_prototypes(train_classes, train_features, all_classes, all_features, distance, prototypes_method, feature_name, dataset_name):
    # build prototypes
    if prototypes_method == "classes":
        # get unique actions
        class_ids = list(set(itertools.chain(*train_classes)))
        class_ids.sort()
        # build mean feature vectors for each action
        prototypes = [
            np.mean([feature for classes, feature in zip(train_classes, train_features) if _class in classes], axis=0)
            for _class in class_ids
        ]
    elif prototypes_method == "kmeans":
        # make clusters from train vectors
        kmeans = KMeans(n_clusters=cfg.SIMILARITY.PROTO.KMEANS.K)
        kmeans.fit(train_features)
        train_predictions = kmeans.predict(train_features)
        prototypes = kmeans.cluster_centers_
        class_ids = range(cfg.SIMILARITY.PROTO.KMEANS.K)
        # tsne = TSNE(2)
        # tsne_res = tsne.fit_transform(train_features)
        # fig = plt.figure(figsize=(20,20))
        # plt.scatter(tsne_res[:,0], tsne_res[:,1], c=train_predictions, cmap='hsv', marker='+')
        # plt.legend()
        # plt.savefig("./figs/kmeans/tsne")

        # import sys; sys.exit()

    # distance to prototypes
    if distance == "euclidean":
        scores = cdist(all_features, prototypes)
    elif distance == "fractional":
        f = cfg.SIMILARITY.FRACTIONAL.F
        scores = cdist(all_features, prototypes, lambda u, v: np.sum(np.abs(u - v)**f)**(1 / f))
    elif distance == "kde":
        assert feature_name in ["hrnet", "ip_csn_152"]
        assert dataset_name == "Memento10k"
        # kde
        scores = []
        for _class in tqdm(class_ids, desc="Generating KDE scores for each class"):
            # all_class_features = [feature for classes, feature in zip(all_classes, all_features) if _class in classes]
            if prototypes_method == "classes":
                train_class_features = np.array([feature for classes, feature in zip(train_classes, train_features) if _class in classes])
            elif prototypes_method == "kmeans":
                train_class_features = np.array([feature for prediction, feature in zip(train_predictions, train_features) if prediction == _class])

            if train_class_features.shape[0] >= cfg.SIMILARITY.PROTO.CLASSES.KDE_THRESH or prototypes_method == "kmeans":
                print(f"{train_class_features.shape[0]} elements in class {_class}")
                class_scores = _get_kde(train_class_features, all_features, feature_name, dataset_name, pickle_suffix=f"_{_class}", verbose=False, save=False if prototypes_method == "kmeans" else True)
                # checking ---------------------------------------------------------------------------
                nb_inf = np.count_nonzero(class_scores == - np.inf) + np.count_nonzero(class_scores == np.inf) + np.count_nonzero(class_scores == np.nan)
                if nb_inf != 0: print(f"Warning! There are {nb_inf} inf/nan values among the KDE scores for class {_class}.")
                clean_class_scores = class_scores[class_scores != - np.inf]
                clean_class_scores = clean_class_scores[clean_class_scores !=   np.inf]
                clean_class_scores = clean_class_scores[clean_class_scores !=   np.nan]
                mean = clean_class_scores.mean()
                class_scores[class_scores == - np.inf] = mean
                class_scores[class_scores ==   np.inf] = mean
                class_scores[class_scores ==   np.nan] = mean
                scores.append(class_scores)
                # print(f"min: {class_scores.min():.2f}, max: {class_scores.max():.2f}")

        scores = np.stack(scores, axis=-1)
    return scores

def _make_similarity(similarity_method, feature_name, dataset_name, use_pca, hrnet_frames=1):
    print(f"\n[SIMILARITY] Making {dataset_name} features for {feature_name} with {similarity_method}")

    root_dir = cfg.DIR.ROOT_DIRS[dataset_name]

    all_filenames = get_all_data(dataset_name)
    train_filenames = get_train_data(dataset_name)
    features_dir = os.path.join(root_dir, FEATURES_INFO[feature_name]["folder_name"])
    if   dataset_name == "Memento10k": origin_extension = ".mp4"
    elif dataset_name == "LaMem":      origin_extension = ".jpg"
    elif dataset_name == "VideoMem":   origin_extension = ".mp4"
    suffix = f"_{hrnet_frames}frames_avg" if feature_name == "hrnet" and dataset_name in ["Memento10k", "VideoMem"] else ""
    if dataset_name == "Memento10k":
        all_filenames   = [record["filename"].replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in all_filenames]
        train_filenames = [record["filename"].replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in train_filenames]
    elif dataset_name == "LaMem":
        all_filenames   = [record[0].replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in all_filenames]
        train_filenames = [record[0].replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in train_filenames]
    elif dataset_name == "VideoMem":
        all_filenames   = [record.replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in all_filenames]
        train_filenames = [record.replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in train_filenames]

    all_features   = np.array([load_array_from_file(os.path.join(features_dir, filename)) for filename in tqdm(all_filenames,   desc="Building full dataset")])
    train_features = np.array([load_array_from_file(os.path.join(features_dir, filename)) for filename in tqdm(train_filenames, desc="Building train dataset")])

    if "." in similarity_method:
        threshold = float(similarity_method[-3:])
        real_similarity_method = similarity_method[:-3]
    else:
        real_similarity_method = similarity_method

    pickle_suffix = ""
    # don't know why but it works better
    if dataset_name == "Memento10k" and real_similarity_method == "kde":
        if feature_name == "meanOF":
            train_features = train_features / 100000
            all_features   = all_features   / 100000
        if feature_name == "hrnet" or feature_name == "ip_csn_152":
            train_features = train_features * 10
            all_features   = all_features   * 10
        pickle_suffix = "kde"

    # reduce to 10 components
    if use_pca and train_features.shape[-1] > cfg.SIMILARITY.PCA.COMP:
        pca = _get_or_make_pickle_object("pca", train_features, feature_name, pca_comp=cfg.SIMILARITY.PCA.COMP, dataset_name=dataset_name, pickle_suffix=pickle_suffix)
        train_features = pca.transform(train_features)
        all_features   = pca.transform(all_features)

    if real_similarity_method == "kde":
        scores = _get_kde(train_features, all_features, feature_name, dataset_name)
    elif real_similarity_method == "cosine_mean":
        scores = _get_cosine(train_features, all_features, submethod="mean")
    elif real_similarity_method == "cosine_threshold":
        scores = _get_cosine(train_features, all_features, submethod="threshold", threshold=threshold)
    elif real_similarity_method == "fractional_mean":
        scores = _get_fractional(train_features, all_features, submethod="mean")
    elif real_similarity_method == "dbscan":
        scores = _get_dbscan(train_features, all_features, feature_name)
    else:
        raise NotImplementedError

    # checking ---------------------------------------------------------------------------
    nb_inf = np.count_nonzero(scores == - np.inf) + np.count_nonzero(scores == np.inf) + np.count_nonzero(scores == np.nan)
    if nb_inf != 0: print(f"Warning! There are {nb_inf} inf/nan values among the KDE scores.")
    print(f"min: {scores.min():.2f}, max: {scores.max():.2f}")

    # saving -----------------------------------------------------------------------------
    for filename, score in zip(all_filenames, scores):
        # suffix = threshold if method == "cosine_threshold" else ""
        path = os.path.join(root_dir, "similarity", similarity_method, FEATURES_INFO[feature_name]["folder_name"], filename.replace(".npy", ".txt"))
        save_array_to_file(np.expand_dims(score, axis=0), path)

def make_kde(feature_name, dataset_name):
    _make_similarity("kde", feature_name, dataset_name, use_pca=True)
def make_cosine_mean(feature_name, dataset_name):
    _make_similarity("cosine_mean", feature_name, dataset_name, use_pca=True)
def make_cosine_threshold(feature_name, dataset_name, threshold):
    _make_similarity("cosine_threshold" + str(threshold), feature_name, dataset_name, use_pca=True)
def make_fractional_mean(feature_name, dataset_name):
    _make_similarity("fractional_mean", feature_name, dataset_name, use_pca=True)
def _make_distance_to_prototypes(feature_name, dataset_name, distance, prototypes_method="classes", use_pca=True, hrnet_frames=1):
    assert dataset_name == "Memento10k"
    similarity_method = distance + "_to_prototypes_"  + prototypes_method
    if distance == "kde" and prototypes_method =="classes": similarity_method += str(cfg.SIMILARITY.PROTO.CLASSES.KDE_THRESH)
    if distance == "kde" and prototypes_method =="kmeans":  similarity_method += str(cfg.SIMILARITY.PROTO.KMEANS.K)
    print(f"\n[SIMILARITY] Making {dataset_name} features for {feature_name} with {similarity_method}")
    root_dir = cfg.DIR.ROOT_DIRS[dataset_name]

    all_records = get_all_data(dataset_name)
    train_records = get_train_data(dataset_name)
    features_dir = os.path.join(root_dir, FEATURES_INFO[feature_name]["folder_name"])
    if   dataset_name == "Memento10k": origin_extension = ".mp4"
    elif dataset_name == "LaMem":      origin_extension = ".jpg"
    suffix = f"_{hrnet_frames}frames_avg" if feature_name == "hrnet" and dataset_name == "Memento10k" else ""
    if dataset_name == "Memento10k":
        all_filenames   = [record["filename"].replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in all_records]
        train_filenames = [record["filename"].replace(origin_extension, f"{suffix}{FEATURES_INFO[feature_name]['file_extension']}") for record in train_records]
        all_classes     = [record["action_labels"] for record in all_records]
        train_classes   = [record["action_labels"] for record in train_records]

    all_features   = np.array([load_array_from_file(os.path.join(features_dir, filename)) for filename in tqdm(all_filenames,   desc="Building full dataset")])
    train_features = np.array([load_array_from_file(os.path.join(features_dir, filename)) for filename in tqdm(train_filenames, desc="Building train dataset")])

    pickle_suffix = ""
    # don't know why but it works better
    if dataset_name == "Memento10k" and distance == "kde":
        if feature_name == "hrnet" or feature_name == "ip_csn_152":
            train_features = train_features * 10
            all_features   = all_features   * 10
        pickle_suffix = "kde"

    # reduce to 10 components
    if use_pca and train_features.shape[-1] > cfg.SIMILARITY.PCA.COMP:
        pca = _get_or_make_pickle_object("pca", train_features, feature_name, pca_comp=cfg.SIMILARITY.PCA.COMP, dataset_name=dataset_name, pickle_suffix=pickle_suffix)
        train_features = pca.transform(train_features)
        all_features   = pca.transform(all_features)

    scores = _get_distance_to_prototypes(train_classes, train_features, all_classes, all_features, distance, prototypes_method, feature_name, dataset_name)

    # saving -----------------------------------------------------------------------------
    for filename, score in zip(all_filenames, scores):
        # suffix = threshold if method == "cosine_threshold" else ""
        path = os.path.join(root_dir, "similarity", similarity_method, FEATURES_INFO[feature_name]["folder_name"], filename)
        save_array_to_file(score, path)
def make_euclidean_to_prototypes(feature_name, dataset_name):
    _make_distance_to_prototypes(feature_name, dataset_name, distance="euclidean")
def make_fractional_to_prototypes(feature_name, dataset_name):
    _make_distance_to_prototypes(feature_name, dataset_name, distance="fractional")
def make_kde_to_prototypes_classes(feature_name, dataset_name):
    _make_distance_to_prototypes(feature_name, dataset_name, distance="kde", prototypes_method="classes")
def make_kde_to_prototypes_kmeans(feature_name, dataset_name):
    _make_distance_to_prototypes(feature_name, dataset_name, distance="kde", prototypes_method="kmeans")
def make_dbscan_clusters(feature_name, dataset_name):
    _make_similarity("dbscan", feature_name, dataset_name, use_pca=True)


if __name__ == '__main__':
    for dataset_name in ["VideoMem", "Memento10k"]:
        # raw perception
        make_contrast(dataset_name)
        make_hog(dataset_name)
        make_brightness(dataset_name)
        make_blurriness(dataset_name)
        make_size_orig(dataset_name)
        # # semantic
        make_hrnet(dataset_name)
        make_csn(dataset_name)
        # similarity
        for feature_name in ["ip_csn_152", "hrnet"]:
            make_dbscan_clusters("hrnet", dataset_name)