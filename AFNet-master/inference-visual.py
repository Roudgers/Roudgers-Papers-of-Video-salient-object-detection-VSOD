#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function
import os

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF
import numpy as np

import argparse
from tqdm import tqdm

from libs.datasets import get_transforms, get_datasets
# from libs.networks import VideoModel
from libs.networks.models_visual import VideoModel
from libs.utils.pyt_utils import load_model
from libs.utils.metric import StructureMeasure

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    ),
    'stage2_cfg': dict(
        NUM_BRANCHES = 2,
        NUM_CHANNELS = [32, 64],
        NUM_BLOCKS = [4, 4],
    ),
    'stage3_cfg': dict(
        NUM_BRANCHES = 3,
        NUM_CHANNELS=[256, 512, 1024],
        NUM_BLOCKS=[4, 4, 4],
    ),
    'stage4_cfg': dict(
        NUM_BRANCHES = 4,
        NUM_BLOCKS = [4, 4, 4, 4],
        NUM_CHANNELS = [256, 512, 1024, 256],
    )
}

CFG = configurations
parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--data', type=str, default='//media/lewis/Win 10 Pro x64/datasets/RCRNet/RCRNet-dynamic-version1/data/datasets/',
                    help='path to datasets folder')
parser.add_argument('--dataset', default='ViSal', type=str, choices=['DAVIS2016', 'VOS', 'ViSal', 'DAVSOD', 'FBMS', 'SegTrack-V2'],
                    help='dataset name for inference')
parser.add_argument('--split', default='test', type=str, choices=['test', 'val'],
                    help='dataset split for inference')
parser.add_argument('--checkpoint', default='models/checkpoints/video_epoch-10.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--results-folder', default='data/results/',
                    help='location to save predicted saliency maps')
parser.add_argument('--dynamic_before', default='data/dynamic_before/',
                    help='location to save dynamic_before maps')
parser.add_argument('--dynamic_after', default='data/dynamic_after/',
                    help='location to save dynamic_after maps')
parser.add_argument('--dynamic_before_rear', default='data/dynamic_before_rear/',
                    help='location to save other_before maps')
parser.add_argument('--dynamic_after_rear', default='data/dynamic_after_rear/',
                    help='location to save other_before maps')
parser.add_argument('--other_before', default='data/other_before/',
                    help='location to save other_before maps')
parser.add_argument('--other_after', default='data/other_after/',
                    help='location to save other_after maps')
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')

# Model settings
parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')
parser.add_argument("--clip_len", type=int, default=4,
                    help="the number of frames in a video clip.")

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

data_transforms = get_transforms(
    input_size=(args.size, args.size),
    image_mode=False
)
dataset = get_datasets(
    name_list=args.dataset,
    split_list=args.split,
    config_path=args.dataset_config,
    root=args.data,
    training=False,
    transforms=data_transforms['test'],
    read_clip=True,
    random_reverse_clip=False,
    label_interval=1,
    frame_between_label_num=0,
    clip_len=args.clip_len
)

dataloader = data.DataLoader(
    dataset=dataset,
    batch_size=1, # only support 1 video clip
    num_workers=args.num_workers,
    shuffle=False
)

model = VideoModel(output_stride=args.os, cfg=CFG)

# load pretrained models
if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    model = load_model(model=model, model_file=args.checkpoint, is_restore=True)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

model.to(device)


def inference():    # _call_函数，将类实例当做函数来使用
    model.eval()
    print("Begin inference on {} {}.".format(args.dataset, args.split))
    running_mae = 0.0
    running_smean = 0.0
    for data in tqdm(dataloader):
        images = [frame['image'].to(device) for frame in data]
        labels = []
        for frame in data:
            # images.append(frame['image'].to(device))
            labels.append(frame['label'].to(device))
        with torch.no_grad():
            preds, f_k2_front_visuals, o_k2_front_visuals, f_k2_rear_visuals, o_k2_rear_visuals = model(images)
            # preds = [torch.sigmoid(pred) for pred in preds]
        # save predicted saliency maps
        for i, (label_, pred_, f_k2_front_visual_, o_k2_front_visual_, f_k2_rear_visual_, o_k2_rear_visual_) in enumerate(zip(labels, preds, f_k2_front_visuals, o_k2_front_visuals, f_k2_rear_visuals, o_k2_rear_visuals)):
            for j, (label, pred, feats_encode, Premask, f_k2_rear_visual,  o_k2_rear_visual) in enumerate(zip(label_.detach().cpu(), pred_.detach().cpu(), f_k2_front_visual_.detach().cpu(), o_k2_front_visual_.detach().cpu(), f_k2_rear_visual_.detach().cpu(),  o_k2_rear_visual_.detach().cpu())):
                # pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                dataset = data[i]['dataset'][j]
                image_id = data[i]['image_id'][j]
                height = data[i]['height'].item()
                width = data[i]['width'].item()
                result_path = os.path.join(args.results_folder, "{}/{}.png".format(dataset, image_id))
                dynamic_before_path = os.path.join(args.dynamic_before, "{}/{}.png".format(dataset, image_id))
                dynamic_after_path = os.path.join(args.dynamic_after, "{}/{}.png".format(dataset, image_id))
                dynamic_before_path_rear = os.path.join(args.dynamic_before_rear, "{}/{}.png".format(dataset, image_id))
                dynamic_after_path_rear = os.path.join(args.dynamic_after_rear, "{}/{}.png".format(dataset, image_id))

                result = TF.to_pil_image(pred)
                # result = result.resize((height, width))
                dirname = os.path.dirname(result_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                result.save(result_path)

                feats_encode_split = torch.chunk(feats_encode, 128, dim=0)
                for k, dynamic_before_res in enumerate(feats_encode_split):
                    dynamic_before = TF.to_pil_image(dynamic_before_res)
                    dynamic_before = dynamic_before.resize((height, width))
                    k = str(k)
                    dynamic_before_path_res = os.path.join(dynamic_before_path, k + ".png")
                    dirname = os.path.dirname(dynamic_before_path_res)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    dynamic_before.save(dynamic_before_path_res)

                Premask_split = torch.chunk(Premask, 128, dim=0)
                for k, Premask_res in enumerate(Premask_split):
                    dynamic_after = TF.to_pil_image(Premask_res)
                    dynamic_after = dynamic_after.resize((height, width))
                    k = str(k)
                    dynamic_after_path_res = os.path.join(dynamic_after_path, k + ".png")
                    dirname = os.path.dirname(dynamic_after_path_res)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    dynamic_after.save(dynamic_after_path_res)

                f_k2_rear_visual_split = torch.chunk(f_k2_rear_visual, 128, dim=0)
                for k, f_k2_rear_visual_res in enumerate(f_k2_rear_visual_split):
                    dynamic_before_rear = TF.to_pil_image(f_k2_rear_visual_res)
                    dynamic_before_rear = dynamic_before_rear.resize((height, width))
                    k = str(k)
                    dynamic_before_path_rear_res = os.path.join(dynamic_before_path_rear, k + ".png")
                    dirname = os.path.dirname(dynamic_before_path_rear_res)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    dynamic_before_rear.save(dynamic_before_path_rear_res)

                o_k2_rear_visual_split = torch.chunk(o_k2_rear_visual, 128, dim=0)
                for k, o_k2_rear_visual_res in enumerate(o_k2_rear_visual_split):
                    dynamic_after_rear = TF.to_pil_image(o_k2_rear_visual_res)
                    dynamic_after_rear = dynamic_after_rear.resize((height, width))
                    k = str(k)
                    dynamic_after_path_rear_res = os.path.join(dynamic_after_path_rear, k + ".png")
                    dirname = os.path.dirname(dynamic_after_path_rear_res)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    dynamic_after_rear.save(dynamic_after_path_rear_res)

                pred_idx = pred[0, :, :].numpy()
                label_idx = label[0, :, :].numpy()

                running_smean += StructureMeasure(pred_idx.astype(np.float32), (label_idx >= 0.5).astype(np.bool))
                running_mae += np.abs(pred_idx - label_idx).mean()
    samples_num = len(dataloader.dataset)
    samples_num *= args.clip_len


    epoch_mae = running_mae / samples_num
    epoch_smeasure = running_smean / samples_num
    print('MAE: {:.4f}'.format(epoch_mae))
    print('S-measure: {:.4f}'.format(epoch_smeasure))

if __name__ == "__main__":
    inference()
