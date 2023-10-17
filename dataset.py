import os
import json
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import skvideo.io
from transforms.feature_transforms import NormalizeRaw, NormalizeSimilarity
from transforms.video_transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip
from config import FEATURES_INFO
from utils import load_array_from_file
from config import cfg
from PIL import Image
import pickle
import csv
class DatasetMemento(Dataset):

    def __init__(self,
        # given in argparse (required)
        split,
        data_augmentation,
        use_raw,
        raw_features,
        normalize_raw,
        compute_raw,
        use_temporal_std,
        fixed_features,
        use_hrnet,
        use_csn,
        hrnet_frames,
        csn_arch,
        use_similarity,
        similarity_methods,
        similarity_features,
        normalize_similarity,
        # optional
        input_transform         = None,
        return_record           = False,
        force_load_input        = False,
        limit_to                = None,
        ):
        print(f"Loading {split} dataset")

        # SETTING ATTRIBUTES =============================================================

        # UTILS --------------------------------------------------------------------------
        self.name                    = "Memento10k"
        self.root_dir                = cfg.DIR.ROOT_DIRS[self.name]
        self.split                   = split
        # INPUT --------------------------------------------------------------------------
        self.input_transform         = self._get_input_transforms(input_transform, data_augmentation)
        # RAW ----------------------------------------------------------------------------
        self.use_raw                 = use_raw
        self.raw_features            = raw_features
        self.raw_transform           = NormalizeRaw(raw_features=raw_features, use_temporal_std=use_temporal_std, dataset=self.name) if normalize_raw else None
        self.use_temporal_std        = use_temporal_std
        self.compute_raw             = compute_raw
        # HRNET & CSN --------------------------------------------------------------------
        self.fixed_features          = fixed_features
        self.use_hrnet               = use_hrnet
        self.use_csn                 = use_csn
        self.hrnet_frames            = hrnet_frames
        self.csn_arch                = csn_arch
        # SIMILARITY --------------------------------------------------------------------
        self.use_similarity         = use_similarity
        self.similarity_methods     = similarity_methods
        self.similarity_features    = similarity_features
        self.similarity_transform   = NormalizeSimilarity(similarity_features=similarity_features, dataset=self.name) if normalize_similarity else None
        # MISC ---------------------------------------------------------------------------
        self.return_record           = return_record
        self.force_load_input        = force_load_input

        # BUILDING DATA ==================================================================
        if split == "all":                      data = self._load_data(["train", "val", "test"])
        elif split == "train_val":              data = self._load_data(["train", "val"])
        elif split in ["train", "val", "test"]: data = self._load_data([split])
        else:                                   raise ValueError("Split not recognized")
        self.data = data
        if limit_to is not None:
            self.data = self.data[:limit_to]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        record = self.data[idx]

        # UTILS -------------------------------------------------------------------
        name = record['filename']
        sample['name'] = name
        sample['url'] = record['url']
        if self.return_record:
            sample['record'] = record

        # MEM_SCORE -------------------------------------------------------------------
        if self.split != "test" and self.split != "all":
            sample['mem_score'] = record['mem_score']

        # INPUT -------------------------------------------------------------------
        if (not self.fixed_features and self.fixed_features is not None or self.force_load_input):
            input_path = os.path.join(self.root_dir, "videos_npy", name.replace(".mp4", ".npy"))
            # print(f"loading {name.replace('.mp4', '.npy')}")
            input = np.load(input_path).astype(np.float32) / 255.
            if self.input_transform is not None:
                input = self.input_transform(input)
            sample['input'] = input


        # RAW FEATURES -------------------------------------------------------------------
        if self.use_raw and not self.compute_raw:
            raw_list = []
            for feature in self.raw_features:
                feature_info = FEATURES_INFO[feature]
                feature_path = os.path.join(
                    self.root_dir,
                    feature_info["folder_name"],
                    name.replace(".mp4", feature_info["file_extension"]))
                assert os.path.isfile(feature_path), f"Raw features have to be computed before training. Cannot find file {feature_path}."
                np_feature = load_array_from_file(feature_path)

                if not self.use_temporal_std:  # only mean of value across time
                    np_feature = np_feature[:,0]
                else:                          # mean and std across time
                    np_feature = np_feature.flatten(order='F')  # order='F' for hog, to have 10 mean THEN 10 std (easier to normalize)

                raw_list.append(np_feature)

            raw = np.concatenate(raw_list, axis=0)

            if self.raw_transform and raw is not None:
                raw = self.raw_transform(raw)
            sample['raw'] = raw

        # HRNET FEATURES -------------------------------------------------------------------
        if self.use_hrnet and self.fixed_features:
            feature_info = FEATURES_INFO["hrnet"]
            feature_path = os.path.join(
                self.root_dir,
                feature_info["folder_name"],
                name.replace(".mp4", f"_{self.hrnet_frames}frames_avg" + feature_info["file_extension"]))
            np_feature = load_array_from_file(feature_path)
            sample["hrnet"] = np_feature

        # CSN FEATURES -------------------------------------------------------------------
        if self.use_csn and self.fixed_features:
            feature_info = FEATURES_INFO[self.csn_arch]
            feature_path = os.path.join(
                self.root_dir,
                feature_info["folder_name"],
                name.replace(".mp4", feature_info["file_extension"]))
            np_feature = load_array_from_file(feature_path)
            sample[self.csn_arch] = np_feature

        # SIMILARITY FEATURES -------------------------------------------------------------------
        if self.use_similarity:
            similarity_list = []
            for method in self.similarity_methods:
                extension = ".npy" if "_to_prototypes" in method else ".txt"
                similarity_list_tmp = []
                for feature in self.similarity_features:
                    feature_info = FEATURES_INFO[feature]
                    suffix = "" if feature != "hrnet" else f"_{self.hrnet_frames}frames_avg"
                    feature_path = os.path.join(
                        self.root_dir,
                        "similarity",
                        method,
                        feature_info["folder_name"],
                        name.replace(".mp4", suffix + extension))
                    assert os.path.isfile(feature_path), f"Similarity scores have to be computed before training. Cannot find file {feature_path}."
                    np_feature = load_array_from_file(feature_path)
                    if "_to_prototypes" not in method:
                        np_feature = np_feature[0]
                    similarity_list_tmp.append(np_feature)

                similarity_tmp = np.concatenate(similarity_list_tmp, axis=0)
                if self.similarity_transform:
                    similarity_tmp = self.similarity_transform(similarity_tmp, method)

                similarity_list.append(similarity_tmp)

            similarity = np.concatenate(similarity_list, axis=0)
            sample['similarity'] = similarity

        return sample

    def _load_data(self, splits):
        data = []
        for _split in splits:
            with open(os.path.join(self.root_dir, f"memento_{_split}_data.json"), 'r') as _data:
                data += json.load(_data)
        return data

    def _get_input_transforms(self, input_transform, data_augmentation):
        if input_transform is not None:
            return input_transform
        # else
        if data_augmentation:
            return transforms.Compose([
                ToTensor(),
                RandomResizedCrop(256, 256, scale=(.8,1.0), aspect_ratio=(1.0,1.0)),
                RandomHorizontalFlip(),
            ])
        # else
        return None


class DatasetLaMem(Dataset):

    def __init__(self,
        # given in argparse (required)
        split,
        data_augmentation,
        use_raw,
        raw_features,
        normalize_raw,
        use_temporal_std,
        fixed_features,
        use_hrnet,
        use_csn,
        hrnet_frames,
        csn_arch,
        use_similarity,
        similarity_methods,
        similarity_features,
        normalize_similarity,
        # optional
        input_transform         = None,
        return_record           = False,
        limit_to                = None,
        ):
        print(f"Loading {split} dataset")

        # SETTING ATTRIBUTES =============================================================

        # UTILS --------------------------------------------------------------------------
        self.name                    = "LaMem"
        self.root_dir                = cfg.DIR.ROOT_DIRS[self.name]
        self.split                   = split
        # INPUT --------------------------------------------------------------------------
        self.input_transform         = self._get_input_transforms(input_transform, data_augmentation)
        # RAW ----------------------------------------------------------------------------
        self.use_raw                 = use_raw
        self.raw_features            = raw_features
        self.raw_transform           = NormalizeRaw(raw_features=raw_features, use_temporal_std=use_temporal_std, dataset=self.name) if normalize_raw else None
        self.use_temporal_std        = use_temporal_std
        # HRNET & CSN --------------------------------------------------------------------
        self.fixed_features          = fixed_features
        self.use_hrnet               = use_hrnet
        self.use_csn                 = use_csn
        self.hrnet_frames            = hrnet_frames
        self.csn_arch                = csn_arch
        assert self.hrnet_frames in [None, 1]
        # SIMILARITY --------------------------------------------------------------------
        self.use_similarity         = use_similarity
        self.similarity_methods     = similarity_methods
        self.similarity_features    = similarity_features
        self.similarity_transform   = NormalizeSimilarity(similarity_features=similarity_features, dataset=self.name) if normalize_similarity else None
        # MISC ---------------------------------------------------------------------------
        self.return_record           = return_record

        # BUILDING DATA ==================================================================
        if split == "all":                      data = self._load_data(["train", "val", "test"])
        elif split == "train_val":              data = self._load_data(["train", "val"])
        elif split in ["train", "val", "test"]: data = self._load_data([split])
        else:                                   raise ValueError("Split not recognized")
        self.data = data
        if limit_to is not None:
            self.data = self.data[:limit_to]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        record = self.data[idx]

        # UTILS -------------------------------------------------------------------
        name = record[0]
        sample['name'] = name
        if self.return_record:
            sample['record'] = record

        # MEM_SCORE -------------------------------------------------------------------
        sample['mem_score'] = float(record[1])

        # INPUT -------------------------------------------------------------------
        if not self.fixed_features and self.fixed_features is not None:
            input_path = os.path.join(self.root_dir, "images", name)
            input = np.array(Image.open(input_path).convert('RGB').resize((256, 256))).astype(np.float32) / 255.  # cubic interpolation
            # print(f"loading {name.replace('.mp4', '.npy')}")
            if self.input_transform is not None:
                input = self.input_transform(input)
            sample['input'] = input


        # RAW FEATURES -------------------------------------------------------------------
        if self.use_raw:
            raw_list = []
            for feature in self.raw_features:
                feature_info = FEATURES_INFO[feature]
                feature_path = os.path.join(
                    self.root_dir,
                    feature_info["folder_name"],
                    name.replace(".jpg", feature_info["file_extension"]))
                assert os.path.isfile(feature_path), f"Raw features have to be computed before training. Cannot find file {feature_path}."
                np_feature = load_array_from_file(feature_path)

                if not self.use_temporal_std:  # only mean of value across time
                    np_feature = np_feature[:,0]
                else:                          # mean and std across time
                    np_feature = np_feature.flatten(order='F')  # order='F' for hog, to have 10 mean THEN 10 std (easier to normalize)

                raw_list.append(np_feature)

            raw = np.concatenate(raw_list, axis=0)

            if self.raw_transform and raw is not None:
                raw = self.raw_transform(raw)
            sample['raw'] = raw

        # HRNET FEATURES -------------------------------------------------------------------
        if self.use_hrnet and self.fixed_features:
            feature_info = FEATURES_INFO["hrnet"]
            feature_path = os.path.join(
                self.root_dir,
                feature_info["folder_name"],
                name.replace(".jpg", feature_info["file_extension"]))
            np_feature = load_array_from_file(feature_path)
            sample["hrnet"] = np_feature

        # CSN FEATURES -------------------------------------------------------------------
        if self.use_csn and self.fixed_features:
            feature_info = FEATURES_INFO[self.csn_arch]
            feature_path = os.path.join(
                self.root_dir,
                feature_info["folder_name"],
                name.replace(".jpg", feature_info["file_extension"]))
            np_feature = load_array_from_file(feature_path)
            sample[self.csn_arch] = np_feature

        # SIMILARITY FEATURES -------------------------------------------------------------------
        if self.use_similarity:
            similarity_list = []
            for method in self.similarity_methods:
                similarity_list_tmp = []
                for feature in self.similarity_features:
                    feature_info = FEATURES_INFO[feature]
                    feature_path = os.path.join(
                        self.root_dir,
                        "similarity",
                        method,
                        feature_info["folder_name"],
                        name.replace(".jpg", ".txt"))
                    assert os.path.isfile(feature_path), f"Similarity scores have to be computed before training. Cannot find file {feature_path}."
                    np_feature = load_array_from_file(feature_path)[0]
                    similarity_list_tmp.append(np_feature)

                similarity_tmp = np.concatenate(similarity_list_tmp, axis=0)
                if self.similarity_transform:
                    similarity_tmp = self.similarity_transform(similarity_tmp, method)

                similarity_list.append(similarity_tmp)

            similarity = np.concatenate(similarity_list, axis=0)
            sample['similarity'] = similarity

        return sample

    def _load_data(self, splits):
        data = []
        for _split in splits:
            data.append(np.array(np.loadtxt(os.path.join(self.root_dir, "splits", _split + '_1.txt'), delimiter=' ', dtype=str)))
        data = np.concatenate(data, axis=0)
        return data

    def _get_input_transforms(self, input_transform, data_augmentation):
        if input_transform is not None:
            return input_transform
        # else
        if data_augmentation:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(256, 256, scale=(.8,1.0), aspect_ratio=(1.0,1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        # else
        return None


class DatasetVideoMem(Dataset):

    def __init__(self,
        # given in argparse (required)
        split,
        data_augmentation,
        use_raw,
        raw_features,
        normalize_raw,
        compute_raw,
        use_temporal_std,
        fixed_features,
        use_hrnet,
        use_csn,
        hrnet_frames,
        csn_arch,
        use_similarity,
        similarity_methods,
        similarity_features,
        normalize_similarity,
        # optional
        input_transform         = None,
        return_record           = False,
        force_load_input        = False,
        limit_to                = None,
        ):
        print(f"Loading {split} dataset")

        # SETTING ATTRIBUTES =============================================================

        # UTILS --------------------------------------------------------------------------
        self.name                    = "VideoMem"
        self.root_dir                = cfg.DIR.ROOT_DIRS[self.name]
        self.split                   = split
        # INPUT --------------------------------------------------------------------------
        self.input_transform         = self._get_input_transforms(input_transform, data_augmentation)
        # RAW ----------------------------------------------------------------------------
        self.use_raw                 = use_raw
        self.raw_features            = raw_features
        self.raw_transform           = NormalizeRaw(raw_features=raw_features, use_temporal_std=use_temporal_std, dataset=self.name) if normalize_raw else None
        self.use_temporal_std        = use_temporal_std
        self.compute_raw             = compute_raw
        # HRNET & CSN --------------------------------------------------------------------
        self.fixed_features          = fixed_features
        self.use_hrnet               = use_hrnet
        self.use_csn                 = use_csn
        self.hrnet_frames            = hrnet_frames
        self.csn_arch                = csn_arch
        # SIMILARITY --------------------------------------------------------------------
        self.use_similarity         = use_similarity
        self.similarity_methods     = similarity_methods
        self.similarity_features    = similarity_features
        self.similarity_transform   = NormalizeSimilarity(similarity_features=similarity_features, dataset=self.name) if normalize_similarity else None
        # MISC ---------------------------------------------------------------------------
        self.return_record           = return_record
        self.force_load_input        = force_load_input

        # BUILDING DATA ==================================================================
        if split == "train_val":        data = self._load_data(["train", "val"])
        elif split in ["train", "val"]: data = self._load_data([split])
        else:                           raise ValueError("Split not recognized")
        self.data = data
        if limit_to is not None:
            self.data = self.data[:limit_to]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        record = self.data[idx]

        # UTILS -------------------------------------------------------------------
        name = record[0]
        sample['name'] = name

        # MEM_SCORE -------------------------------------------------------------------
        sample['mem_score'] = np.array(record[1]).astype(np.float32)

        # INPUT -------------------------------------------------------------------
        if (not self.fixed_features and self.fixed_features is not None or self.force_load_input):
            input_path = os.path.join(self.root_dir, "resized_mp4", name)
            # print(f"loading {name.replace('.mp4', '.npy')}")
            input = skvideo.io.vread(input_path)[:,:,99:355,:].astype(np.float32) / 255.
            if self.input_transform is not None:
                input = self.input_transform(input)
            sample['input'] = input


        # RAW FEATURES -------------------------------------------------------------------
        if self.use_raw and not self.compute_raw:
            raw_list = []
            for feature in self.raw_features:
                feature_info = FEATURES_INFO[feature]
                feature_path = os.path.join(
                    self.root_dir,
                    feature_info["folder_name"],
                    name.replace(".mp4", feature_info["file_extension"]))
                assert os.path.isfile(feature_path), f"Raw features have to be computed before training. Cannot find file {feature_path}."
                np_feature = load_array_from_file(feature_path)

                if not self.use_temporal_std:  # only mean of value across time
                    np_feature = np_feature[:,0]
                else:                          # mean and std across time
                    np_feature = np_feature.flatten(order='F')  # order='F' for hog, to have 10 mean THEN 10 std (easier to normalize)

                raw_list.append(np_feature)

            raw = np.concatenate(raw_list, axis=0)

            if self.raw_transform and raw is not None:
                raw = self.raw_transform(raw)
            sample['raw'] = raw

        # HRNET FEATURES -------------------------------------------------------------------
        if self.use_hrnet and self.fixed_features:
            feature_info = FEATURES_INFO["hrnet"]
            feature_path = os.path.join(
                self.root_dir,
                feature_info["folder_name"],
                name.replace(".mp4", f"_{self.hrnet_frames}frames_avg" + feature_info["file_extension"]))
            np_feature = load_array_from_file(feature_path)
            sample["hrnet"] = np_feature

        # CSN FEATURES -------------------------------------------------------------------
        if self.use_csn and self.fixed_features:
            feature_info = FEATURES_INFO[self.csn_arch]
            feature_path = os.path.join(
                self.root_dir,
                feature_info["folder_name"],
                name.replace(".mp4", feature_info["file_extension"]))
            np_feature = load_array_from_file(feature_path)
            sample[self.csn_arch] = np_feature

        # SIMILARITY FEATURES -------------------------------------------------------------------
        if self.use_similarity:
            similarity_list = []
            for method in self.similarity_methods:
                extension = ".npy" if "_to_prototypes" in method else ".txt"
                similarity_list_tmp = []
                for feature in self.similarity_features:
                    feature_info = FEATURES_INFO[feature]
                    suffix = "" if feature != "hrnet" else f"_{self.hrnet_frames}frames_avg"
                    feature_path = os.path.join(
                        self.root_dir,
                        "similarity",
                        method,
                        feature_info["folder_name"],
                        name.replace(".mp4", suffix + extension))
                    assert os.path.isfile(feature_path), f"Similarity scores have to be computed before training. Cannot find file {feature_path}."
                    np_feature = load_array_from_file(feature_path)
                    if "_to_prototypes" not in method:
                        np_feature = np_feature[0]
                    similarity_list_tmp.append(np_feature)

                similarity_tmp = np.concatenate(similarity_list_tmp, axis=0)
                if self.similarity_transform:
                    similarity_tmp = self.similarity_transform(similarity_tmp, method)

                similarity_list.append(similarity_tmp)

            similarity = np.concatenate(similarity_list, axis=0)
            sample['similarity'] = similarity

        # print(sample)
        # for a in sample:
        #     print(a, type(sample[a]))
        return sample

    def _load_data(self, splits):
        data = []
        # filenames
        filenames_list = []
        with open(os.path.join(self.root_dir, "train_test_split_videomem.pkl"), 'rb') as f:
            filenames_tuple = pickle.load(f)
        split_idxs = {"train": 0, "val": 1}
        for split in splits:
            filenames = filenames_tuple[split_idxs[split]]
            filenames = [filename.replace(".webm", ".mp4") for filename in filenames]
            filenames_list.append(np.array(filenames))
        filenames_list = np.concatenate(filenames_list, axis=0)
        # mem_scores
        mem_score_dict = {}
        with open(os.path.join(self.root_dir, "ground-truth_dev-set.csv"), 'r') as f:
            csv_file = csv.reader(f)
            for i, line in enumerate(csv_file):
                if i == 0: continue
                mem_score_dict[line[0]] = float(line[1])
        mem_scores = np.array([mem_score_dict[filename.replace(".mp4", ".webm")] for filename in filenames_list]).astype(np.float32)

        data = np.stack([filenames_list, mem_scores], axis=1)

        return data

    def _get_input_transforms(self, input_transform, data_augmentation):
        if input_transform is not None:
            return input_transform
        # else
        if data_augmentation:
            return transforms.Compose([
                ToTensor(),
                RandomResizedCrop(256, 256, scale=(.8,1.0), aspect_ratio=(1.0,1.0)),
                RandomHorizontalFlip(),
            ])
        # else
        return None
