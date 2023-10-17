# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# models
from model.hrnet import build_hrnet
from vmz.models import ip_csn_152, ir_csn_152
from config import FEATURES_INFO, SIMILARITY_INFO
from transforms.feature_transforms import NormalizeRaw, NormalizeSimilarity



vmz_models = {
    "ip_csn_152": ip_csn_152,
    "ir_csn_152": ir_csn_152,
}

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class M3S_MLP(nn.Module):

    def __init__(self, in_dim, hidden_channels):
        super(M3S_MLP, self).__init__()

        layer_dims = [in_dim] + hidden_channels + [1]
        self.num_layers = len(layer_dims) - 1

        for i in range(self.num_layers):
            is_final = True if (i == self.num_layers - 1) else False
            linear   = self._make_block(layer_dims[i], layer_dims[i + 1], is_final=is_final)
            setattr(self, f"linear_{i}", linear)
        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_dim, out_dim, is_final):
        if is_final:
            return nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            return nn.Sequential(nn.Linear(in_dim, out_dim), Mish())

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f"linear_{i}")(x)
        x = self.sigmoid(x)

        return x



def get_hrnet():
    return build_hrnet(decoder_arch="custom_downsample_to_1")

def get_csn(csn_arch):
    return vmz_models[csn_arch](pretraining="ig_ft_kinetics_32frms")

class M3S(nn.Module):
    def __init__(self,
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
        similarity_features,
        similarity_methods,
        hidden_channels,
        dataset_name,
        frozen=True,
        ):
        super(M3S, self).__init__()

        # DECOMPOSE MODEL
        self.use_raw          = use_raw
        self.use_temporal_std = use_temporal_std
        self.use_csn          = use_csn
        self.use_hrnet        = use_hrnet
        self.use_similarity  = use_similarity

        # RAW FEATURES
        self.raw_features = raw_features
        self.normalize_raw = normalize_raw
        self.raw_transform = NormalizeRaw(raw_features=raw_features, use_temporal_std=use_temporal_std, dataset=dataset_name) if normalize_raw else None


        # FIXED FEATURES OR NOT
        self.fixed_features = fixed_features
        if not self.fixed_features:
            if self.use_hrnet:
                self.hrnet_frames = hrnet_frames
                self.hrnet = get_hrnet()
            if self.use_csn:
                self.csn   = get_csn(csn_arch=csn_arch)

        # SIMILARITY
        self.similarity_methods  = similarity_methods
        self.similarity_features = similarity_features

        # MLP
        raw_dim   = sum([FEATURES_INFO[f]["nb_features"] * (1 + FEATURES_INFO[f]["has_temporal_std"] * self.use_temporal_std) for f in raw_features]) if use_raw else 0  # HOG has 10 elements
        hrnet_dim = use_hrnet       * 720
        csn_dim   = use_csn         * 2048
        comp_dim  = sum([SIMILARITY_INFO[m]["nb_features"] for m in similarity_methods]) if use_similarity else 0
        comp_dim *= len(similarity_features) if use_similarity else 0
        self.mlp = M3S_MLP(
            in_dim=int(raw_dim + hrnet_dim + csn_dim + comp_dim),
            hidden_channels=hidden_channels,
        )

        # FREEZING
        if frozen and not self.fixed_features and (self.use_hrnet or self.use_csn):
            self.freeze_core()

        # MISC
        self.dataset_name = dataset_name



    def forward(self, input, raw=None, csn=None, hrnet=None, similarity=None):
        x_list = []

        # PREPARING FEATURES =============================================================

        # RAW FEATURES -------------------------------------------------------------------
        if self.use_raw:
            if raw is None:  # if compute_raw
                raw = self._compute_raw(input)
            x_list.append(raw)

        # HRNET FEATURES -----------------------------------------------------------------
        if self.use_hrnet:
            if hrnet is None:  # if not fixed_features
                hrnet = self._compute_hrnet(input)
            x_list.append(hrnet)

        # CSN FEATURES -------------------------------------------------------------------
        if self.use_csn:
            if csn is None:    # if not fixed_features
                csn = self._compute_csn(input)
            x_list.append(csn)

        # SIMILARITY FEATURES -----------------------------------------------------------
        if self.use_similarity:
            assert similarity is not None
            x_list.append(similarity)

        # FORWARD ========================================================================
        x = torch.cat(x_list, dim=1)
        x = self.mlp(x)

        return x

    def _slice_hrnet_input(self, input):
        """Makes it easy to implement other types of slicing (first-last, etc)"""
        if self.hrnet_frames in [1, 5, 9]:
            input_hrnet = input[:,::input.shape[1] // self.hrnet_frames,:,:,:]
        else:
            raise NotImplementedError
        return input_hrnet

    # def _compute_raw(self, input):
    #     device = input.get_device()
    #     input = np.array(input.cpu())
    #     batch_raw = []
    #     for batch_elt in input:  # batch
    #         raw_list = []
    #         for feature_name in self.raw_features:
    #             feature_fun = getattr(low_level, f"_get_{feature_name}")
    #             if feature_name == "hog":
    #                 pca_name = f"pca_hog.pickle"
    #                 pca_path = os.path.join(cfg.DIR.PICKLE, self.dataset_name, pca_name)
    #                 assert os.path.isfile(pca_path)
    #                 with open(pca_path, 'rb') as f:
    #                     pca = pickle.load(f)
    #             feature = np.array([feature_fun(frame) for frame in batch_elt])
    #             if feature_name == "hog":
    #                 feature = pca.transform(feature)

    #             np_feature = np.stack([feature.mean(axis=0), feature.std(axis=0)], axis=1)
    #             if not self.use_temporal_std:  # only mean of value across time
    #                 np_feature = np_feature[:,0]
    #             else:                          # mean and std across time
    #                 np_feature = np_feature.flatten(order='F')  # order='F' for hog, to have 10 mean THEN 10 std (easier to normalize)

    #             raw_list.append(np_feature)

    #         raw = np.concatenate(raw_list, axis=0)

    #         if self.raw_transform is not None:
    #             raw = self.raw_transform(raw)

    #         batch_raw.append(raw)
    #     raw = np.stack(batch_raw, axis=0)
    #     raw = torch.from_numpy(raw).float().to(device)
    #     return raw

    # def _compute_hrnet(self, input):
    #     input_hrnet = self._slice_hrnet_input(input)
    #     hrnet = [self.hrnet(input_hrnet[:,i,:,:,:].permute(0, 3, 1, 2)).squeeze(-1).squeeze(-1) for i in range(input_hrnet.shape[1])]
    #     hrnet = torch.stack(hrnet).mean(axis=0)
    #     return hrnet

    # def _compute_csn(self, input):
    #     csn   = self.csn(input.permute(0, 4, 1, 2, 3)).squeeze(-1).squeeze(-1).squeeze(-1)
    #     return csn

    def freeze_core(self):
        print("Freezing model core")
        for param in self.hrnet.parameters(): param.requires_grad = False
        for param in self.csn.parameters():   param.requires_grad = False

    def unfreeze_core(self):
        print("Unfreezing model core")
        for param in self.hrnet.parameters(): param.requires_grad = True
        for param in self.csn.parameters():   param.requires_grad = True

