"""Training script."""
import os
import argparse
import random
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
# utils
import wandb
from tqdm import tqdm
# custom
from config import cfg
from model.m3s import M3S
from dataset import DatasetLaMem, DatasetMemento, DatasetVideoMem
from utils import ic, set_seeds, draw_hist
from loss import get_loss, kl_divergence, js_divergence


# ARGPARSE ===============================================================================
parser = argparse.ArgumentParser(description="Training script for memorability model")
parser.add_argument("-n", "--name",           default="X",                                                                                           help="name to append to run id")
parser.add_argument("-s", "--save",           default=False,                         action="store_true",                                            help="whether saving weights")
parser.add_argument("-l", "--log",            default=False,                         action="store_true",                                            help="whether logging in WANDB")
# parser.add_argument("-i", "--log_images",     default=False,                         action="store_true",                                            help="whether logging images in WANDB")
parser.add_argument("--seed",                 default=cfg.TRAIN.SEED,                type=int,                                                       help="random seed, will be set for numpy and pytorch")
parser.add_argument("--group",                default=None,                                                                                          help="to group runs in wandb")
parser.add_argument("--tags",                 default=[],                                                 nargs='+',                                 help="tags in wandb")
# INPUT ----------------------------------------------------------------------------------
parser.add_argument("--data_augmentation",    default=cfg.TRAIN.DATA_AUGMENTATION,   action="store_true",                                            help="whether using data augmentation for input video, requires fixed_features set to False")
# RAW ------------------------------------------------------------------------------------
parser.add_argument("--use_raw",              default=cfg.TRAIN.USE_RAW,             action="store_true",                                            help="whether using the raw module")
parser.add_argument("--raw_features",         default=cfg.TRAIN.RAW_FEATURES,                             nargs='+',                                 help="list of raw features")
parser.add_argument("--normalize_raw",        default=cfg.TRAIN.NORMALIZE_RAW,       action="store_true",                                            help="whether normalizing them")
parser.add_argument("--compute_raw",          default=cfg.TRAIN.COMPUTE_RAW,         action="store_true",                                            help="force computation of raw features")
parser.add_argument("--use_temporal_std",     default=cfg.TRAIN.USE_TEMPORAL_STD,    action="store_true",                                            help="whether using the temporal std of the raw features")
# HRNET & CSN ----------------------------------------------------------------------------
parser.add_argument("--fixed_features",       default=cfg.TRAIN.FIXED_FEATURES,      action="store_true",                                            help="whether using pre-computed features")
parser.add_argument("--use_hrnet",            default=cfg.TRAIN.USE_HRNET,           action="store_true",                                            help="whether using the HRNet module")
parser.add_argument("--use_csn",              default=cfg.TRAIN.USE_CSN,             action="store_true",                                            help="whether using the CSN module")
parser.add_argument("--hrnet_frames",         default=cfg.TRAIN.HRNET_FRAMES,        type=int,            choices=cfg.CONST.HRNET_FRAMES_LIST,       help="nb frames as input to HRNet")
parser.add_argument("--csn_arch",             default=cfg.TRAIN.CSN_ARCH,                                 choices=cfg.CONST.CSN_ARCH_LIST,           help="CSN architecture")
# SIMILARITY ----------------------------------------------------------------------------
parser.add_argument("--use_similarity",      default=cfg.TRAIN.USE_SIMILARITY,     action="store_true",                                            help="whether using the similarity module")
parser.add_argument("--similarity_features", default=cfg.TRAIN.SIMILARITY_FEATURES,                     nargs='+',                                 help="list of similarity features")
parser.add_argument("--similarity_methods",  default=cfg.TRAIN.SIMILARITY_METHODS,                      nargs='+',                                 help="list of similarity methods")
parser.add_argument("--normalize_similarity",default=cfg.TRAIN.NORMALIZE_SIMILARITY,action="store_true",                                           help="whether normalizing them")
# MLP ------------------------------------------------------------------------------------
parser.add_argument("--hidden_channels",      default=cfg.TRAIN.HIDDEN_CHANNELS,     type=int,            nargs='+',                                 help="MLP hidden layers")
# TRAINING -------------------------------------------------------------------------------
parser.add_argument("--dataset",              default=cfg.TRAIN.DATASET,                                  choices=cfg.CONST.DATASET_LIST,            help="dataset on which to train")
parser.add_argument("--batch_size",           default=cfg.TRAIN.BATCH_SIZE,          type=int,                                                       help="batch size")
parser.add_argument("--loss",                 default=cfg.TRAIN.LOSS,                                     choices=cfg.CONST.LOSS_LIST,               help="loss function")
parser.add_argument("--mse_tails_pow",        default=cfg.TRAIN.MSETAILS.POW,        type=int,                                                       help="")
parser.add_argument("--mse_tails_mult_fact",  default=cfg.TRAIN.MSETAILS.MULT_FACT,  type=float,                                                     help="")
parser.add_argument("--lr",                   default=cfg.TRAIN.LEARNING_RATE,       type=float,                                                     help="start learning rate")
parser.add_argument("--epochs",               default=cfg.TRAIN.EPOCHS,              type=int,                                                       help="nb of epochs")
parser.add_argument("--optimizer",            default=cfg.TRAIN.OPTIMIZER,                                                                           help="optimizer name")
parser.add_argument("--weight_decay",         default=cfg.TRAIN.WEIGHT_DECAY,        type=float,                                                     help="optimizer weight decay (regularization)")
parser.add_argument("--scheduler",            default=cfg.TRAIN.SCHEDULER,                                                                           help="scheduler name")
parser.add_argument("--scheduler_gamma",      default=cfg.TRAIN.SCHEDULER_GAMMA,     type=float,                                                     help="coeff for learning rate decay")
parser.add_argument("--scheduler_step_size",  default=cfg.TRAIN.SCHEDULER_STEP_SIZE, type=int,                                                       help="step for lerning rate decay")
parser.add_argument("--checkpoint_local" ,    default=cfg.TRAIN.CHECKPOINT_LOCAL,    action="store_true",                                            help="using local storage or wandb cloud for loading checkpoint")
parser.add_argument("--checkpoint_run_id",    default=cfg.TRAIN.CHECKPOINT_RUN_ID,                                                                   help="ID of checkpoint run in wandb")
parser.add_argument("--checkpoint_epoch",     default=cfg.TRAIN.CHECKPOINT_EPOCH,    type=int,                                                       help="epoch of checkpoint run in wandb")
args = parser.parse_args()


# GETTING RUN VARIABLES ==================================================================
name                 = args.name
save                 = args.save
log                  = args.log
# log_images           = args.log_images
date                 = cfg.CONST.DATE
seed                 = args.seed if args.seed is not None else random.randint(1, 2 ** 32)
run_id               = date + "_" + name
dataset              = args.dataset
data_augmentation    = args.data_augmentation
use_raw              = args.use_raw
raw_features         = args.raw_features
normalize_raw        = args.normalize_raw
compute_raw          = args.compute_raw
use_temporal_std     = args.use_temporal_std
fixed_features       = args.fixed_features
use_hrnet            = args.use_hrnet
use_csn              = args.use_csn
hrnet_frames         = args.hrnet_frames
csn_arch             = args.csn_arch
use_similarity       = args.use_similarity
similarity_features  = args.similarity_features
similarity_methods   = args.similarity_methods
normalize_similarity = args.normalize_similarity
hidden_channels      = args.hidden_channels
batch_size           = args.batch_size
loss                 = args.loss
mse_tails_pow        = args.mse_tails_pow
mse_tails_mult_fact  = args.mse_tails_mult_fact
lr                   = args.lr
epochs               = args.epochs
optimizer            = args.optimizer
weight_decay         = args.weight_decay
scheduler            = args.scheduler
scheduler_gamma      = args.scheduler_gamma
scheduler_step_size  = args.scheduler_step_size
checkpoint_local     = args.checkpoint_local
checkpoint_run_id    = args.checkpoint_run_id
checkpoint_epoch     = args.checkpoint_epoch
device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
group                = args.group
tags                 = args.tags


# POST-PROCESSING ARGS ===================================================================
# loss
if loss == "spearman" and batch_size <= 2:
    print(f"Warning: loss is Spearman RC and batch_size is {batch_size}. Please increase batch_size.")
# fixed features
if not fixed_features:
    print(f"Warning: Scripts heven't been optimized for chen fixed_features is set to False.")
    import sys; sys.exit()
if data_augmentation and fixed_features:
    print(f"Warning: Data augmentation is only relevant when fixed_features is set to False. Please change one of the 2 parameters.")
    import sys; sys.exit()
if use_similarity and not fixed_features:
    print(f"Warning: Similarity is only relevant when fixed_features is set to True. Please change one of the 2 parameters.")
    import sys; sys.exit()
# setting some irrelevant parameters to None
if not use_raw:                raw_features   = None; normalize_raw = None
if not use_hrnet:              hrnet_frames   = None
if not use_csn:                csn_arch       = None
if not (use_hrnet or use_csn): fixed_features = None
if not use_similarity:        similarity_methods = None; similarity_features = None; normalize_similarity = None
if use_similarity:
    if "hrnet"      in similarity_features: hrnet_frames = args.hrnet_frames
    if "ip_csn_152" in similarity_features: csn_arch = args.csn_arch
# setting some relevant parameters to True
# if save and log:               log_images = True
# if log_images:                 log        = True
# checking that some parameters are present
if use_raw:                    assert raw_features         not in [None, []]
if use_similarity:            assert similarity_features not in [None, []]
if use_similarity:            assert similarity_methods  not in [None, []]
if loss != "mse_tails": mse_tails_pow = None; mse_tails_mult_fact = None

ic(args)
ic(name)
ic(device)


set_seeds(seed)

# TRAINING OBJECTS =======================================================================
# dataset
dataset_kwargs = {
    "use_raw"             : use_raw,
    "use_hrnet"           : use_hrnet,
    "use_csn"             : use_csn,
    "use_similarity"      : use_similarity,
    "raw_features"        : raw_features,
    "normalize_raw"       : normalize_raw,
    "compute_raw"         : compute_raw,
    "use_temporal_std"    : use_temporal_std,
    "fixed_features"      : fixed_features,
    "csn_arch"            : csn_arch,
    "hrnet_frames"        : hrnet_frames,
    "similarity_features": similarity_features,
    "similarity_methods" : similarity_methods,
    "normalize_similarity":normalize_similarity,
}
if dataset == "Memento10k": dataset_class = DatasetMemento
elif dataset == "LaMem":    dataset_class = DatasetLaMem
elif dataset == "VideoMem": dataset_class = DatasetVideoMem
else:                       raise NotImplementedError
train_dataset    = dataset_class(split="train", data_augmentation=data_augmentation, **dataset_kwargs)
val_dataset      = dataset_class(split="val",   data_augmentation=False,             **dataset_kwargs)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True,  batch_size=batch_size, num_workers=cfg.CONST.NUM_WORKERS)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   shuffle=False, batch_size=batch_size, num_workers=cfg.CONST.NUM_WORKERS)

# model and optimizer
model = M3S(
    use_raw=use_raw,
    raw_features=raw_features,
    normalize_raw=normalize_raw,
    use_temporal_std=use_temporal_std,
    fixed_features=fixed_features,
    use_hrnet=use_hrnet,
    use_csn=use_csn,
    hrnet_frames=hrnet_frames,
    csn_arch=csn_arch,
    use_similarity=use_similarity,
    similarity_features=similarity_features,
    similarity_methods=similarity_methods,
    hidden_channels=hidden_channels,
    dataset_name=dataset,
).to(device)


if checkpoint_run_id is not None:
    assert checkpoint_epoch is not None
    if checkpoint_local:  # local
        weights = torch.load(os.path.join(cfg.DIR.WEIGHTS), checkpoint_run_id, f"e_{checkpoint_epoch}.pts")
        model.load_state_dict(weights)
    else:  # wandb
        weights = wandb.restore(f"weights/e_{checkpoint_epoch}.pts", run_path=f"{cfg.CONST.WANDB_USERNAME}/{cfg.CONST.WANDB_PROJECT}/{checkpoint_run_id}")
        weights = torch.load(weights.name)
        model.load_state_dict(weights)

loss_kwargs = {
    "mse_tails_pow": mse_tails_pow,
    "mse_tails_mult_fact": mse_tails_mult_fact,
}
criterion = get_loss(loss, **loss_kwargs)
mae_criterion = torch.nn.L1Loss()
mse_criterion = torch.nn.MSELoss()
optimizer = getattr(torch.optim, optimizer)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
scheduler = getattr(torch.optim.lr_scheduler, scheduler)(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma, verbose=False)

# wandb
if log:
    wandb.init(project=cfg.CONST.WANDB_PROJECT, entity=cfg.CONST.WANDB_USERNAME, group=group, tags=tags)
    config = wandb.config
    config.run_id               = run_id
    config.device               = device
    config.save                 = save
    config.seed                 = seed
    config.data_augmentation    = data_augmentation
    config.use_raw              = use_raw
    config.raw_features         = raw_features
    config.normalize_raw        = normalize_raw
    config.compute_raw          = compute_raw
    config.use_temporal_std     = use_temporal_std
    config.use_hrnet            = use_hrnet
    config.use_csn              = use_csn
    config.hrnet_frames         = hrnet_frames
    config.csn_arch             = csn_arch
    config.fixed_features       = fixed_features
    config.use_similarity       = use_similarity
    config.similarity_features  = similarity_features
    config.similarity_methods   = similarity_methods
    config.normalize_similarity = normalize_similarity
    config.hidden_channels      = hidden_channels
    config.batch_size           = batch_size
    config.lr                   = lr
    config.epochs               = epochs
    config.scheduler_gamma      = scheduler_gamma
    config.scheduler_step_size  = scheduler_step_size
    config.dataset              = train_dataset.name
    config.model                = model.__class__.__name__
    config.loss                 = loss
    config.criterion            = criterion.__class__.__name__
    config.weight_decay         = weight_decay
    config.optimizer            = optimizer.__class__.__name__
    config.scheduler            = scheduler.__class__.__name__
    config.num_workers          = cfg.CONST.NUM_WORKERS
    config.checkpoint_local     = checkpoint_local
    config.checkpoint_run_id    = checkpoint_run_id
    config.checkpoint_epoch     = checkpoint_epoch
    wandb.watch(model, log_freq=1)


# TRAINING ===============================================================================
best_rho = 0

for epoch in range(epochs):

    # TRAINING ---------------------------------------------------------------------------
    model.train()
    batch_losses = []

    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - training  ")

    for i, sample in enumerate(loop):

        input       = sample['input'].float().to(device)       if not fixed_features and fixed_features is not None else None
        raw         = sample['raw'].float().to(device)         if use_raw else None
        csn         = sample[csn_arch].float().to(device)      if fixed_features and use_csn   else None
        hrnet       = sample['hrnet'].float().to(device)       if fixed_features and use_hrnet else None
        similarity = sample["similarity"].float().to(device) if use_similarity else None
        mem_score   = sample['mem_score'].unsqueeze(1).float().to(device)

        output = model(input, raw, csn, hrnet, similarity)

        alpha = epoch / (epochs - 1) if epochs > 1 else 0
        loss = criterion(output, mem_score, alpha=alpha)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.cpu().item())

        loop.set_postfix(loss=loss.cpu().item(), out=output[0,0].cpu().item())

        if i == len(loop) - 1:
            training_epoch_loss = np.mean(batch_losses)
            loop.set_postfix(loss=training_epoch_loss, out=output[0,0].cpu().item())

    if log:
        wandb.log({"train_loss": training_epoch_loss}, step=epoch)

    # VALIDATION -------------------------------------------------------------------------
    model.eval()
    batch_losses = []
    batch_maes = []
    batch_mses = []

    scores_gt = []
    scores_pr = []
    table_data = []

    loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - validation")
    with torch.no_grad():
        for i, sample in enumerate(loop):

            input       = sample['input'].float().to(device)       if (not fixed_features and fixed_features is not None) else None
            raw         = sample['raw'].float().to(device)         if use_raw else None
            csn         = sample[csn_arch].float().to(device)      if fixed_features and use_csn   else None
            hrnet       = sample['hrnet'].float().to(device)       if fixed_features and use_hrnet else None
            similarity = sample["similarity"].float().to(device) if use_similarity else None

            mem_score   = sample['mem_score'].unsqueeze(1).float().to(device)

            output = model(input, raw, csn, hrnet, similarity)

            alpha = epoch / (epochs - 1) if epochs > 1 else 0
            loss = criterion(output, mem_score, alpha=alpha)
            mae = mae_criterion(output, mem_score)
            mse = mse_criterion(output, mem_score)
            batch_losses.append(loss.cpu().item())
            batch_maes.append(mae.cpu().item())
            batch_mses.append(mse.cpu().item())

            loop.set_postfix(corr='?', loss=loss.cpu().item(), out=output[0,0].cpu().item())

            scores_gt += mem_score.tolist()
            scores_pr += output.tolist()

            if log and epoch == epochs - 1:
                mse = (mem_score - output) ** 2
                mem_score = mem_score.cpu()
                output    = output.cpu()
                mse       = mse.cpu()
                mse_worst, id_worst = torch.max(mse, dim=0)
                mse_best,  id_best  = torch.min(mse, dim=0)
                table_data.append([i, "worst", sample["name"][id_worst], mem_score[id_worst,0].item(), output[id_worst,0].item(), mse[id_worst,0].item()])
                table_data.append([i, "best",  sample["name"][id_best],  mem_score[id_best,0].item(),  output[id_best,0].item(),  mse[id_best,0].item()])

            if i == len(loop) - 1:
                scores_gt = np.array(scores_gt)
                scores_pr = np.array(scores_pr)
                val_epoch_loss = np.mean(batch_losses)
                val_epoch_mae  = np.mean(batch_maes)
                val_epoch_mse  = np.mean(batch_mses)
                rho, _ = spearmanr(scores_gt, scores_pr)
                tau, _ = kendalltau(scores_gt, scores_pr)
                kl     = kl_divergence(scores_pr, scores_gt)
                js     = js_divergence(scores_pr, scores_gt)
                loop.set_postfix(corr=rho, loss=val_epoch_loss, out=output[0,0].cpu().item())

        if rho > best_rho:
            best_rho = rho

        if log:
            # distribution histogram
            hist_gt, hist_pr, fig = draw_hist(scores_gt, scores_pr)

            wandb.log({
                "val_spearman" : rho,
                "val_kendall"  : tau,
                "val_loss"     : val_epoch_loss,
                "val_mae"      : val_epoch_mae,
                "val_mse"      : val_epoch_mse,
                "lr"           : optimizer.param_groups[0]['lr'],
                "val_dist_gt"  : wandb.Histogram(np_histogram=hist_gt),
                "val_dist_pred": wandb.Histogram(np_histogram=hist_pr),
                "val_dist_kl"  : kl,
                "val_dist_js"  : js,
                "val_dist_both": wandb.Image(fig),
                }, step=epoch)
            wandb.run.summary["best_spearman"] = best_rho
            if epoch == epochs - 1:
                val_table = wandb.Table(data=table_data, columns=["Batch", "Type", "Name", "Ground truth", "Prediction", "MSE"])
                wandb.log({
                    "val_table"    : val_table,
                    }, step=epoch)

    scheduler.step()

    if save:
        # wandb
        if log:
            WANDB_PATH = os.path.join(wandb.run.dir, "weights")
            if not os.path.isdir(WANDB_PATH):
                os.mkdir(WANDB_PATH)
            torch.save(model.state_dict(), os.path.join(WANDB_PATH, f"e_{epoch}.pts"))
            wandb.save("*.pts")
        # local
        WEIGHTS_PATH = os.path.join(cfg.DIR.WEIGHTS, run_id)
        if not os.path.isdir(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, f"e_{epoch}.pts"))

