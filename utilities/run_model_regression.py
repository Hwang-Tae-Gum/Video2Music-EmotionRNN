import torch
import time

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr
import torch.nn.functional as F


def pcc_loss(pred, target):
    """1 - Pearson Correlation Coefficient. 모델이 평균만 예측하면 PCC=0 → loss=1로 페널티."""
    p = pred.flatten()
    t = target.flatten()
    p_c = p - p.mean()
    t_c = t - t.mean()
    pcc = (p_c * t_c).sum() / (p_c.norm() * t_c.norm() + 1e-8)
    return 1.0 - pcc


def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1, norm_stats=None):
    nd_mean = norm_stats["nd_mean"] if norm_stats else 0.0
    nd_std  = norm_stats["nd_std"]  if norm_stats else 1.0
    lv_mean = norm_stats["lv_mean"] if norm_stats else 0.0
    lv_std  = norm_stats["lv_std"]  if norm_stats else 1.0

    out = -1
    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()
        opt.zero_grad()

        feature_semantic_list = []
        for feature_semantic in batch["semanticList"]:
            feature_semantic_list.append(feature_semantic.to(get_device()))

        feature_scene_offset = batch["scene_offset"].to(get_device())
        feature_motion       = batch["motion"].to(get_device())
        feature_emotion      = batch["emotion"].to(get_device())

        feature_note_density = batch["note_density"].to(get_device())
        feature_loudness     = batch["loudness"].to(get_device())

        y = model(feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion)
        y = y.reshape(y.shape[0] * y.shape[1], -1)

        feature_loudness     = feature_loudness.flatten().reshape(-1, 1)
        feature_note_density = feature_note_density.flatten().reshape(-1, 1)

        # z-score 정규화
        feature_note_density_n = (feature_note_density - nd_mean) / nd_std
        feature_loudness_n     = (feature_loudness     - lv_mean) / lv_std

        y_nd, y_lv = torch.split(y, 1, dim=1)

        loss_density  = F.mse_loss(y_nd, feature_note_density_n) + 0.5 * pcc_loss(y_nd, feature_note_density_n)
        loss_loudness = F.mse_loss(y_lv, feature_loudness_n)     + 0.5 * pcc_loss(y_lv, feature_loudness_n)

        out = loss_density + loss_loudness
        out.backward()
        opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        time_after = time.time()
        if (batch_num + 1) % print_modulus == 0:
            print(SEPERATOR)
            print("Epoch", cur_epoch, " Batch", batch_num + 1, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(out))
            print("Time (s):", time_after - time_before)
            print(SEPERATOR)
            print("")
    return


def eval_model(model, dataloader, loss, norm_stats=None):
    nd_mean = norm_stats["nd_mean"] if norm_stats else 0.0
    nd_std  = norm_stats["nd_std"]  if norm_stats else 1.0
    lv_mean = norm_stats["lv_mean"] if norm_stats else 0.0
    lv_std  = norm_stats["lv_std"]  if norm_stats else 1.0

    model.eval()

    avg_rmse              = -1
    avg_loss              = -1
    avg_rmse_note_density = -1
    avg_rmse_loudness     = -1

    with torch.set_grad_enabled(False):
        n_test = len(dataloader)

        sum_loss              = 0.0
        sum_rmse              = 0.0
        sum_rmse_note_density = 0.0
        sum_rmse_loudness     = 0.0

        for batch in dataloader:
            feature_semantic_list = []
            for feature_semantic in batch["semanticList"]:
                feature_semantic_list.append(feature_semantic.to(get_device()))

            feature_scene_offset = batch["scene_offset"].to(get_device())
            feature_motion       = batch["motion"].to(get_device())
            feature_emotion      = batch["emotion"].to(get_device())
            feature_loudness     = batch["loudness"].to(get_device())
            feature_note_density = batch["note_density"].to(get_device())

            y = model(feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion)
            y = y.reshape(y.shape[0] * y.shape[1], -1)

            feature_loudness     = feature_loudness.flatten().reshape(-1, 1)
            feature_note_density = feature_note_density.flatten().reshape(-1, 1)

            # loss는 정규화 공간에서
            feature_note_density_n = (feature_note_density - nd_mean) / nd_std
            feature_loudness_n     = (feature_loudness     - lv_mean) / lv_std
            feature_combined_n     = torch.cat((feature_note_density_n, feature_loudness_n), dim=1)

            out = loss.forward(y, feature_combined_n)
            sum_loss += float(out)

            rmse = torch.sqrt(F.mse_loss(y, feature_combined_n))
            sum_rmse += float(rmse)

            y_nd, y_lv = torch.split(y, 1, dim=1)

            # RMSE는 원래 스케일로 역변환
            y_nd_raw = y_nd * nd_std + nd_mean
            y_lv_raw = y_lv * lv_std + lv_mean

            sum_rmse_note_density += float(torch.sqrt(F.mse_loss(y_nd_raw, feature_note_density)))
            sum_rmse_loudness     += float(torch.sqrt(F.mse_loss(y_lv_raw, feature_loudness)))

        avg_loss              = sum_loss  / n_test
        avg_rmse              = sum_rmse  / n_test
        avg_rmse_note_density = sum_rmse_note_density / n_test
        avg_rmse_loudness     = sum_rmse_loudness     / n_test

    return avg_loss, avg_rmse, avg_rmse_note_density, avg_rmse_loudness
