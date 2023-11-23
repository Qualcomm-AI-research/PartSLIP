# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import numpy as np
from torchvision.ops import box_iou
import torch


def normalize_pc(pc_file, save_dir, io, device, save_normalized_pc=False):
    pc = io.load_pointcloud(pc_file, device=device)
    xyz = pc.points_padded().reshape(-1, 3)
    rgb = pc.features_padded().reshape(-1, 3)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / torch.norm(xyz, dim=1, p=2).max().item()
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    if save_normalized_pc:
        save_colored_pc(os.path.join(save_dir, "normalized_pc.ply"), xyz, rgb)
    return xyz, rgb


def save_colored_pc(file_name, xyz, rgb):
    n = xyz.shape[0]
    if rgb.max() < 1.1:
        rgb = (rgb * 255).astype(np.uint8)
    f = open(file_name, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % n)
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    for i in range(n):
        f.write(
            "%f %f %f %d %d %d\n"
            % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][1], rgb[i][2])
        )


def get_iou(bb1: dict, bb2: dict) -> float:
    """
    Get Intersection over Union (IoU) of 2 bounding boxes.

    Args:
        bb1: with keys 'x1', 'x2', 'y1', 'y2', where (x1, y1) and (x2, y2) are the top-left and bottom-right corners
        bb2: same keys as bb1
    """
    assert bb1["x1"] < bb1["x2"] + 1e-6
    assert bb1["y1"] < bb1["y2"] + 1e-6
    assert bb2["x1"] < bb2["x2"] + 1e-6
    assert bb2["y1"] < bb2["y2"] + 1e-6

    keys = ['x1', 'y1', 'x2', 'y2']
    iou = box_iou(
        torch.tensor([bb1[k] for k in keys])[None, :], 
        torch.tensor([bb2[k] for k in keys])[None, :]
    ).item()

    return iou
