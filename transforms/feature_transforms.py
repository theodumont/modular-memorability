import numpy as np
from config import FEATURES_INFO, FEATURES_NORM, SIMILARITY_NORM

class NormalizeRaw(object):

    def __init__(self, raw_features, use_temporal_std, dataset):
        self.raw_features = raw_features
        self.use_temporal_std = use_temporal_std
        self.dataset = dataset

    def __call__(self, raw):
        mean = []
        std =  []
        for feature in self.raw_features:
            mean     += FEATURES_NORM[self.dataset][feature]["mean"]                      # mean of value
            if self.use_temporal_std and FEATURES_INFO[feature]["has_temporal_std"]:
                mean += FEATURES_NORM[self.dataset][feature]["temporal_std_mean"]         # mean of std

            std      += FEATURES_NORM[self.dataset][feature]["std"]                       # std of value
            if self.use_temporal_std and FEATURES_INFO[feature]["has_temporal_std"]:
                std  += FEATURES_NORM[self.dataset][feature]["temporal_std_std"]          # std of std
        mean = np.array(mean)
        std =  np.array(std)
        assert len(mean) == raw.shape[0]
        assert len(std)  == raw.shape[0]
        raw = normalize(raw, mean=mean, std=std)
        return raw

class NormalizeSimilarity(object):

    def __init__(self, similarity_features, dataset):
        self.similarity_features = similarity_features
        self.dataset = dataset

    def __call__(self, similarity, method):
        mean = []
        std = []
        for feature in self.similarity_features:
            mean += SIMILARITY_NORM[self.dataset][method][feature]["mean"]
            std  += SIMILARITY_NORM[self.dataset][method][feature]["std"]
        mean = np.array(mean)
        std =  np.array(std)
        assert len(mean) == similarity.shape[0]
        assert len(std)  == similarity.shape[0]
        similarity = normalize(similarity, mean=mean, std=std)
        return similarity

def normalize(features, mean, std):
    return (features - mean) / std