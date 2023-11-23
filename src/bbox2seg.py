# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import numpy as np

from src.utils import get_iou, save_colored_pc


def check_pc_within_bbox(x1, y1, x2, y2, pc):
    flag = np.logical_and(pc[:, 0] > x1, pc[:, 0] < x2)
    flag = np.logical_and(flag, pc[:, 1] > y1)
    flag = np.logical_and(flag, pc[:, 1] < y2)
    return flag


def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))


def get_union(f, x):
    if f[x] == x:
        return x
    f[x] = get_union(f, f[x])
    return f[x]


def calc_components_connectivity(xyz, components, thr=0.02):
    n = len(components)
    connectivity = np.zeros((n, n), dtype=bool)
    x_min, x_max, y_min, y_max, z_min, z_max = [], [], [], [], [], []
    for i in range(n):
        x_min.append(xyz[components[i], 0].min())
        y_min.append(xyz[components[i], 1].min())
        z_min.append(xyz[components[i], 2].min())
        x_max.append(xyz[components[i], 0].max())
        y_max.append(xyz[components[i], 1].max())
        z_max.append(xyz[components[i], 2].max())

    for i in range(n):
        for j in range(n):
            if (
                x_min[i] > x_max[j] + thr
                or x_max[i] < x_min[j] - thr
                or y_min[i] > y_max[j] + thr
                or y_max[i] < y_min[j] - thr
                or z_min[i] > z_max[j] + thr
                or z_max[i] < z_min[j] - thr
            ):
                continue
            connectivity[i, j] = True
    return connectivity


def bbox2seg(
    xyz,
    superpoint,
    preds,
    screen_coor_all,
    point_idx_all,
    part_names,
    save_dir,
    num_view=10,
    solve_instance_seg=True,
    visualize=True,
):
    print("semantic segmentation...")
    n_category = len(part_names)
    n_sp = len(superpoint)
    sp_visible_cnt = np.zeros(n_sp)  # visible points for each superpoint
    sp_bbox_visible_cnt = np.zeros(
        (n_category, n_sp)
    )  # visible points of superpoint j that are covered by a bounding box of category i
    preds_per_view = [[] for i in range(num_view)]
    for pred in preds:
        preds_per_view[pred["image_id"]].append(pred)
    in_box_ratio_list = [
        [[] for j in range(n_sp)] for i in range(n_category)
    ]  # used for instance segmentation
    for i in range(num_view):
        screen_coor = screen_coor_all[i]  # 2D projected location of each 3D point
        point_idx = point_idx_all[i]  # point index of each 2D pixel
        visible_pts = np.unique(point_idx)[1:]  # the first one is -1
        for k, sp in enumerate(superpoint):
            sp_visible_pts = intersection(sp, visible_pts)
            sp_visible_cnt[k] += len(sp_visible_pts)
            in_bbox = np.zeros((n_category, len(sp_visible_pts)), dtype=bool)

            for pred in preds_per_view[i]:
                cat_id = pred["category_id"] - 1
                x1, y1, w, h = pred["bbox"]
                x2, y2 = x1 + w, y1 + h
                if (
                    check_pc_within_bbox(x1, y1, x2, y2, screen_coor).mean() > 0.98
                ):  # ignore bbox covering the whole objects
                    continue
                if len(sp_visible_pts) == 0:
                    in_box_ratio_list[cat_id][k].append(-1)
                else:
                    mask = check_pc_within_bbox(x1, y1, x2, y2, screen_coor[sp_visible_pts])
                    in_bbox[cat_id] = np.logical_or(in_bbox[cat_id], mask)
                    in_box_ratio_list[cat_id][k].append(mask.mean())
            for j in range(n_category):
                sp_bbox_visible_cnt[j, k] += in_bbox[j].sum()

    sem_score = np.zeros((n_category, n_sp))
    sem_seg = np.ones(xyz.shape[0], dtype=np.int32) * -1
    for j in range(n_category):
        for i in range(n_sp):
            if sp_visible_cnt[i] == 0:
                sem_score[j, i] = 0
            else:
                sem_score[j, i] = sp_bbox_visible_cnt[j, i] / sp_visible_cnt[i]
    for i in range(n_sp):
        if sem_score[:, i].max() < 0.5:
            continue
        idx = -1
        for j in reversed(range(n_category)):  # give priority to small parts
            if sem_score[j, i] >= 0.5 and part_names[j] in [
                "handle",
                "button",
                "wheel",
                "knob",
                "switch",
                "bulb",
                "shaft",
                "touchpad",
                "camera",
                "screw",
            ]:
                idx = j
                break
        if idx == -1:
            idx = np.argmax(sem_score[:, i])
        sem_seg[superpoint[i]] = idx
    if visualize:
        os.makedirs("%s/semantic_seg" % save_dir, exist_ok=True)
        for j in range(n_category):
            rgb_sem = np.ones((xyz.shape[0], 3)) * (sem_seg == j).reshape(-1, 1)
            save_colored_pc("%s/semantic_seg/%s.ply" % (save_dir, part_names[j]), xyz, rgb_sem)

    if solve_instance_seg == False:
        return sem_seg, None

    os.makedirs("%s/instance_seg" % save_dir, exist_ok=True)
    print("instance segmentation...")
    connectivity = calc_components_connectivity(xyz, superpoint)
    ins_seg = np.ones(xyz.shape[0], dtype=np.int32) * -1
    ins_cnt = 0
    for j in range(n_category):
        f = []
        for i in range(n_sp):
            f.append(i)
        for i in range(n_sp):
            if sem_seg[superpoint[i][0]] == j:
                for k in range(i):
                    if sem_seg[superpoint[k][0]] == j and connectivity[i][k]:
                        ratio_i = np.array(in_box_ratio_list[j][i])
                        ratio_k = np.array(in_box_ratio_list[j][k])
                        mask = np.logical_and(ratio_i > -0.5, ratio_k > -0.5)
                        if mask.sum() == 0 or max(ratio_i[mask].sum(), ratio_k[mask].sum()) < 1e-3:
                            dis = 1
                        else:
                            dis = np.abs(ratio_i[mask] - ratio_k[mask]).sum()
                            dis /= max(ratio_i[mask].sum(), ratio_k[mask].sum())
                        l1 = len(superpoint[i])
                        l2 = len(superpoint[k])
                        if dis < 0.1 and max(l1, l2) / min(l1, l2) < 100:
                            f[get_union(f, i)] = get_union(f, k)
        instances = []
        flags = []
        for i in range(n_sp):
            if get_union(f, i) == i and sem_seg[superpoint[i][0]] == j:
                instance = []
                for k in range(n_sp):
                    if get_union(f, k) == i:
                        instance += superpoint[k]
                instances.append(instance)
                flags.append(False)

        # filter out instances that have small iou with bounding boxes
        for i in range(num_view):
            screen_coor = screen_coor_all[i]  # 2D projected location of each 3D point
            point_idx = point_idx_all[i]  # point index of each 2D pixel
            visible_pts = np.unique(point_idx)[1:]  # the first one is -1
            for k, instance in enumerate(instances):
                if flags[k]:
                    continue
                ins_visible_pts = intersection(instance, visible_pts)
                if len(ins_visible_pts) == 0:
                    continue
                ins_coor = screen_coor[ins_visible_pts]
                bb1 = {
                    "x1": ins_coor[:, 0].min(),
                    "y1": ins_coor[:, 1].min(),
                    "x2": ins_coor[:, 0].max(),
                    "y2": ins_coor[:, 1].max(),
                }
                for pred in preds_per_view[i]:
                    cat_id = pred["category_id"] - 1
                    if cat_id != j:
                        continue
                    x1, y1, w, h = pred["bbox"]
                    bb2 = {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}
                    if get_iou(bb1, bb2) > 0.5:
                        flags[k] = True
                        break
        rgb_ins = np.zeros((xyz.shape[0], 3))
        for i in range(len(instances)):
            if flags[i]:
                ins_seg[instances[i]] = ins_cnt
                ins_cnt += 1
                rgb_ins[instances[i]] = np.random.rand(3)
        if visualize:
            save_colored_pc("%s/instance_seg/%s.ply" % (save_dir, part_names[j]), xyz, rgb_ins)
    return sem_seg, ins_seg
