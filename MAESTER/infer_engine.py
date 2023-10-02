import argparse
import os
import time
import torch
import sys
from model import *
from torch.nn import functional as F
import numpy as np



def define_embed_idx(side_patch_num, central_patch_num):
    total_patch_num = int(side_patch_num**2)
    flattern_idx = torch.arange(total_patch_num)
    reshape_idx = flattern_idx.reshape(side_patch_num, side_patch_num)
    mid_point = int(side_patch_num // 2)
    mid_centeral_num = int(central_patch_num // 2)
    selected_idx = reshape_idx[
        mid_point - mid_centeral_num: mid_point + mid_centeral_num,
        mid_point - mid_centeral_num: mid_point + mid_centeral_num,
    ]

    return selected_idx.flatten()


def define_idx(central_patch_num, pix_size):
    index_h = torch.zeros(central_patch_num * central_patch_num)
    index_w = torch.zeros(central_patch_num * central_patch_num)
    count = 0
    for res_h in range(central_patch_num):
        for res_w in range(central_patch_num):
            index_h[count] = res_h * pix_size
            index_w[count] = res_w * pix_size
            count += 1
    return index_h, index_w


def place_res(
        res_stack, target_tensor, anchor_h, anchor_w, index_d, index_h, index_w,
        embed_idx):
    # exclude cls token and only select the central patch
    res_stack = res_stack[:, 1:][:, embed_idx.long()]
    col_num = res_stack.shape[0]

    h = index_h + anchor_h
    h = h.repeat(col_num, 1)
    w = index_w + anchor_w
    w = w.repeat(col_num, 1)

    d = index_d.repeat_interleave(res_stack.shape[1])
    d = d.reshape(col_num, -1)

    target_tensor[d.long(), h.long(), w.long(), :] = res_stack.cpu()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def run_inference(
    rank,
    ngpus_per_node,
    scr,
    cfg,
    model,
    embedding_storage_path,
    device
):
    orgD, orgH, orgW = scr.shape
    central_patch = cfg["MODEL"]["central_patch"]
    target_size = cfg["DATASET"]["vol_size"]
    vol_size = (
        cfg["DATASET"]["vol_size"]
        + int(cfg["DATASET"]["vol_size"] // cfg["DATASET"]["patch_size"])
        if cfg["DATASET"]["patch_size"] % 2 == 0
        else cfg["DATASET"]["vol_size"]
    )
    pix_size = (
        cfg["DATASET"]["patch_size"] + 1
        if cfg["DATASET"]["patch_size"] % 2 == 0
        else cfg["DATASET"]["patch_size"]
    )
    patch_size = cfg["DATASET"]["patch_size"]
    OFFSET = patch_size // 2
    PAD = int(
        OFFSET
        + (
            (((target_size / patch_size) - central_patch) // 2)
            * (patch_size + 1 if patch_size % 2 == 0 else patch_size)
        )
    ) + pix_size // 2

    scr = F.pad(scr, [PAD, PAD, PAD, PAD, 0, 0], "constant", 0).to(device)

    t1 = time.time()
    print("preparing tensor storage...")
    feature_storage = torch.FloatTensor(
        torch.from_file(
            embedding_storage_path,
            shared=True,
            size=orgD * (orgH+PAD) * (orgW+PAD) * cfg["MODEL"]["embed"],
        )
    ).reshape(orgD, orgH+PAD, orgW+PAD, cfg["MODEL"]["embed"])

    _, adjH, adjW = scr.shape

    with torch.no_grad():
        index_h, index_w = define_idx(central_patch, pix_size)
        iter_list = list(split(range(0, orgD), ngpus_per_node))[rank]
        embed_idx = define_embed_idx(target_size // patch_size, patch_size)

        # print(f"rank {rank} ====> getting partition: [{iter_list[0]}, {iter_list[-1]}]")

        iter_list = torch.tensor(iter_list)
        d_list = iter_list.split(int(cfg["DATASET"]["batch_size"] * 2))

        _win = vol_size // patch_size
        h_patch = np.floor(adjH - _win + 1) - _win
        w_patch = np.floor(adjW - _win + 1) - _win
        total_patch = int((h_patch * w_patch + 2 * vol_size + 2 * _win) / _win)

        for b, i_d in enumerate(d_list):
            print(f"\ninferring batch #{b + 1} of {len(d_list)}")
            h_count = 0
            i_h = 0
            patch_count = 1

            while i_h < adjH - vol_size + 1:
                i_w = 0
                w_count = 0

                while i_w < adjW - vol_size + 1:
                    print(f"\t patch #{patch_count} of ~ {total_patch}", end="\r")
                    sample = scr[
                        i_d, i_h: i_h + vol_size, i_w: i_w + vol_size
                    ].to(device)
                    sample = sample.unsqueeze(1)
                    rep = model.infer_latent(sample)[:, 1:, :]
                    place_res(
                        rep, feature_storage, i_h, i_w, i_d, index_h, index_w, embed_idx
                    )
                    w_count += 1

                    del sample
                    del rep

                    if (
                        w_count == pix_size
                        and i_w + central_patch * pix_size - pix_size
                        < adjW - vol_size + 1
                    ):
                        i_w += central_patch * pix_size - pix_size
                        w_count = 0
                        if h_count == 0:
                            patch_count += 1
                    else:
                        i_w += 1
                h_count += 1
                if (
                    h_count == pix_size
                    and i_h + central_patch * pix_size - pix_size < adjH - vol_size + 1
                ):
                    i_h += central_patch * pix_size - pix_size
                    h_count = 0
                    if w_count == 0:
                        patch_count += 1
                else:
                    i_h += 1

            total_patch = patch_count

    print(f"\n\nRank {rank}: Inference finished in {(time.time() - t1) / 60: .2f} minutes.")
