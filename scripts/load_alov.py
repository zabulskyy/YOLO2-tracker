import os
from os import path as osp
import torch
import matplotlib.pyplot as plt
from PIL import Image


def load_alov(dataset_path="/data/alov"):
    groundtruth_path = osp.join(dataset_path, 'groundtruth')
    dataset = dict()
    for folder in sorted(os.listdir(dataset_path)):
        if folder == 'groundtruth': continue
        for f in sorted(os.listdir(osp.join(dataset_path, folder))):
            _path = osp.join(dataset_path, folder, f)
            _gt_path = osp.join(groundtruth_path, folder, f + ".ann")
            dataset[f] = (sorted([osp.join(_path, x) for x in os.listdir(_path)]), _gt_path)
    return dataset


def file2tensor(file):
    with open(file, 'r') as f:
        lines = torch.tensor([[float(y) for y in x.split()] for x in f.readlines()])
        return lines


def tensor_to_bb(gt, gt_idx):
    gt_bb = gt[gt[:, 0] <= gt_idx + 1][-1]
    index, gt_bb = gt_bb[0].tolist(), gt_bb[1:].tolist()
    return index, gt_bb
