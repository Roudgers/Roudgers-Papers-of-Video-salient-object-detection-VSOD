
from __future__ import absolute_import, division, print_function
import os

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image

import argparse
from tqdm import tqdm

from libs.datasets import get_transforms, get_datasets
from libs.networks import VideoModel,ImageModel
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
parser.add_argument('--data', type=str, default='/home/oip/Lau/datasets/',
                    help='path to datasets folder')
parser.add_argument('--dataset', default='ViSal', type=str, choices=['DAVIS2016', 'VOS', 'ViSal', 'DAVSOD', 'FBMS', 'SegTrack-V2'],
                    help='dataset name for inference')
parser.add_argument('--split', default='test', type=str, choices=['test', 'val'],
                    help='dataset split for inference')
parser.add_argument('--checkpoint', default='models/checkpoints/image_current_best_model.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--results-folder', default='data/results/',
                    help='location to save predicted saliency maps')
parser.add_argument('--dynamic_before', default='data/dynamic_before/',
                    help='location to save dynamic_before maps')
parser.add_argument('--dynamic_after', default='data/dynamic_after/',
                    help='location to save dynamic_after maps')
parser.add_argument('--fixation_att', default='data/fixation_att/',
                    help='location to save fixation_att maps')
parser.add_argument('--fixation_after', default='data/fixation_after/',
                    help='location to save do fixation_after maps')
parser.add_argument('--bu1', default='data/bu1/',
                    help='location to save do fixation_after maps')
parser.add_argument('--bu2', default='data/bu2/',
                    help='location to save do fixation_after maps')
parser.add_argument('--bu3', default='data/bu3/',
                    help='location to save do fixation_after maps')
parser.add_argument('--block1', default='data/block1/',
                    help='location to save do fixation_after maps')
parser.add_argument('--block2', default='data/block2/',
                    help='location to save do fixation_after maps')
parser.add_argument('--block3', default='data/block3/',
                    help='location to save do fixation_after maps')
parser.add_argument('--block4', default='data/block4/',
                    help='location to save do fixation_after maps')
parser.add_argument('--Imagemodel-results-folder', default='data/Imagemodel-results/',
                    help='location to save predicted saliency maps')
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')

# Model settings
parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')
parser.add_argument("--clip_len", type=int, default=1,
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
    read_clip=False,
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

model = ImageModel(cfg=CFG, pretrained=True)
# Fmap_fixation_att = args.fixation_att


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
        images = data['image'].to(device)
        labels = data['label'].to(device)

        with torch.no_grad():
            preds = model(images)
            preds = torch.sigmoid(preds)    # 因为做criterion_bce的时候，做了sigmoid,所以这里也要做了再保存
        # save predicted saliency maps
        # for j, (label, pred) in enumerate(zip(labels.detach().cpu(), preds.detach().cpu()):
        for j, (label, pred) in enumerate(zip(labels.detach().cpu(), preds.detach().cpu())):
            dataset = data['dataset'][j]
            image_id = data['image_id'][j]
            height = data['height'].item()
            width = data['width'].item()
            result_path = os.path.join(args.Imagemodel_results_folder, "{}/{}.png".format(dataset, image_id))
            # fixation_att_path = os.path.join(args.fixation_att, "{}/{}.png".format(dataset, image_id))
            # fixation_after_path = os.path.join(args.fixation_after, "{}/{}.png".format(dataset, image_id))
            # bu1_path = os.path.join(args.bu1, "{}/{}/".format(dataset, image_id))
            # bu2_path = os.path.join(args.bu2, "{}/{}/".format(dataset, image_id))
            # bu3_path = os.path.join(args.bu3, "{}/{}/".format(dataset, image_id))
            # block1_path = os.path.join(args.block1, "{}/{}/".format(dataset, image_id))
            # block2_path = os.path.join(args.block2, "{}/{}/".format(dataset, image_id))
            # block3_path = os.path.join(args.block3, "{}/{}/".format(dataset, image_id))
            # block4_path = os.path.join(args.block4, "{}/{}/".format(dataset, image_id))

            result = TF.to_pil_image(pred)
            result = result.resize((height, width))
            dirname = os.path.dirname(result_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            result.save(result_path)


            # bu1 = bu1.numpy()
            # for k,bu1i in enumerate(bu1):
            #     # bu1i = TF.to_pil_image(bu1i)
            #     # bu1i = bu1i.resize((448, 448))
            #     dirname = os.path.dirname(bu1_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     im = Image.fromarray(bu1i)
            #     name = str(k)+".png"
            #     im.convert('RGB').save(dirname+name)
            #     # bu1i.save(bu1_path, str(bu1i))

            # bu1 = torch.chunk(bu1, 128, dim=0)
            # for k, bu1_res in enumerate(bu1):
            #     bu1 = TF.to_pil_image(bu1_res)
            #     bu1 = bu1.resize((height, width))
            #     dirname = os.path.dirname(bu1_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     bu1_path_res = os.path.join(bu1_path, k+".png")
            #     bu1.save(bu1_path_res)
            #
            # bu2 = torch.chunk(bu2, 128, dim=0)
            # for k, bu2_res in enumerate(bu2):
            #     bu2 = TF.to_pil_image(bu2_res)
            #     bu2 = bu2.resize((height, width))
            #     dirname = os.path.dirname(bu2_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     bu2_path_res = os.path.join(bu2_path, k + ".png")
            #     bu2.save(bu2_path_res)
            #
            # bu3 = torch.chunk(bu3, 128, dim=0)
            # for k, bu3_res in enumerate(bu3):
            #     bu3 = TF.to_pil_image(bu3_res)
            #     bu3 = bu3.resize((height, width))
            #     dirname = os.path.dirname(bu3_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     bu3_path_res = os.path.join(bu3_path, k + ".png")
            #     bu3.save(bu3_path_res)
            #
            # block1 = torch.squeeze(block1, dim=0)
            # block1 = torch.chunk(block1, 256, dim=0)
            # for k, block1_res in enumerate(block1):
            #     block1 = TF.to_pil_image(block1_res)
            #     block1 = block1.resize((height, width))
            #     dirname = os.path.dirname(block1_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     block1_path_res = os.path.join(block1_path, k + ".png")
            #     block1.save(block1_path_res)
            #
            # block2 = torch.squeeze(block2, dim=0)
            # block2 = torch.chunk(block2, 512, dim=0)
            # for k, block2_res in enumerate(block2):
            #     block2 = TF.to_pil_image(block2_res)
            #     block2 = block2.resize((height, width))
            #     dirname = os.path.dirname(block2_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     block2_path_res = os.path.join(block2_path, k + ".png")
            #     block2.save(block2_path_res)
            #
            # block3 = torch.squeeze(block3, dim=0)
            # block3 = torch.chunk(block3, 1024, dim=0)
            # for k, block3_res in enumerate(block3):
            #     block3 = TF.to_pil_image(block3_res)
            #     block3 = block3.resize((height, width))
            #     dirname = os.path.dirname(block3_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     block3_path_res = os.path.join(block3_path, k + ".png")
            #     block3.save(block3_path_res)
            #
            # block4 = torch.squeeze(block4, dim=0)
            # block4 = torch.chunk(block4, 256, dim=0)
            # for k, block4_res in enumerate(block4):
            #     block4 = TF.to_pil_image(block4_res)
            #     block4 = block4.resize((height, width))
            #     dirname = os.path.dirname(block4_path)
            #     if not os.path.exists(dirname):
            #         os.makedirs(dirname)
            #     k = str(k)
            #     block4_path_res = os.path.join(block4_path, k + ".png")
            #     block4.save(block4_path_res)
            #
            # fixation_after = fixation_after[1]
            # # fixation_att = torch.unsqueeze(fixation_att, 0)
            # fixation_after = TF.to_pil_image(fixation_after)
            # fixation_after = fixation_after.resize((56, 56))
            # dirname = os.path.dirname(fixation_after_path)
            # if not os.path.exists(dirname):
            #     os.makedirs(dirname)
            # fixation_after.save(fixation_after_path)

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
