import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from skimage.filters import threshold_otsu


device = torch.device('cuda:0')


def load_config_file(config_file):

    all_params = json.load(open(config_file))
    print(all_params)

    return all_params


def min_max_normalize(x):
    min_x = x.min()
    max_x = x.max()
    if (max_x - min_x) == 0:
        return x - min_x
    else:
        return (x - min_x) / (max_x - min_x)


def get_anchor_loss(pos_logits, neg_logits,
    pos_gt, neg_gt, pos_ratio, neg_ratio, naming='anc'):

    pos_log_probs = F.logsigmoid(pos_logits)
    neg_log_probs = F.logsigmoid(neg_logits)

    loss_pos = (- pos_gt * pos_log_probs).mean() * pos_ratio 
    loss_neg = (- neg_gt * neg_log_probs).mean() * neg_ratio

    loss = loss_pos + loss_neg

    loss_details = {
        '{}_pos'.format(naming): loss_pos.item(),
        '{}_neg'.format(naming): loss_neg.item(),
        '{}_sum'.format(naming): loss.item(),
    }

    return loss, loss_details


def get_diffusion_loss(feature_map_1, feature_map_2, 
    logits_1, logits_2, fg_margin, bg_margin, 
    fg_ratio, bg_ratio, naming='diff'):

    probs_1 = torch.sigmoid(logits_1).permute(0, 3, 1, 2)
    probs_2 = torch.sigmoid(logits_2).permute(0, 3, 1, 2)

    size = (feature_map_1.shape[2], feature_map_1.shape[3])

    probs_1 = F.interpolate(probs_1, 
        size=size, mode='bicubic', align_corners=True)

    probs_2 = F.interpolate(probs_2, 
        size=size, mode='bicubic', align_corners=True)

    fg_1 = (probs_1 * feature_map_1).sum(2).sum(2) / (probs_1.sum(2).sum(2) + 1e-10)
    fg_2 = (probs_2 * feature_map_2).sum(2).sum(2) / (probs_2.sum(2).sum(2) + 1e-10)
    
    bg_1 = ((1 - probs_1) * feature_map_1).sum(2).sum(2) / ((1 - probs_1).sum(2).sum(2) + 1e-10)
    bg_2 = ((1 - probs_2) * feature_map_2).sum(2).sum(2) / ((1 - probs_2).sum(2).sum(2) + 1e-10)

    sim = nn.CosineSimilarity(dim=1)

    fg_diff = sim(fg_1, fg_2) - sim(fg_1, bg_1) / 2 - sim(fg_2, bg_2) / 2
    bg_diff = sim(bg_1, bg_2) - sim(fg_1, bg_1) / 2 - sim(fg_2, bg_2) / 2

    loss_fg = F.relu(fg_margin - fg_diff).mean() * fg_ratio
    loss_bg = F.relu(bg_margin - bg_diff).mean() * bg_ratio

    loss = loss_fg + loss_bg

    loss_details = {
        '{}_fg'.format(naming): loss_fg.item(),
        '{}_bg'.format(naming): loss_bg.item(),
        '{}_sum'.format(naming): loss.item(),
    }

    return loss, loss_details


def get_threshold(pred):

    try:
        threshold = threshold_otsu(pred)
    except ValueError:  # all same values
        threshold = pred.mean()

    return threshold


def get_mask(pred, threshold=None, percentile=None, lower_bound=None):

    if (threshold is None) and (percentile is None):
        threshold = get_threshold(pred) 

    elif (threshold is None) and (percentile is not None):
        threshold = np.percentile(pred, percentile)

    elif (threshold is not None) and (percentile is None):
        pass
    else:
        raise Exception('Get Mask Error.')

    if lower_bound:
        threshold = max(threshold, lower_bound)

    return pred > threshold


def get_iou(pred, gt):

    pred = pred > 0
    gt = gt > 0

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    return (intersection + 1e-15) / (union + 1e-15)      


def get_dice(pred, gt):

    pred = pred > 0
    gt = gt > 0

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    return (2 * intersection + 1e-15) / (union + 1e-15) 


##################################################################

def cat_frames_to_video(in_files, out_file, fps):

    subprocess.call([
        'ffmpeg', '-framerate', str(fps), 
        '-i', in_files, 
        '-vf', 'scale=480:270',
        out_file
    ])


def cat_two_videos_horizontal(in_file1, in_file2, out_file):

    subprocess.call([
        'ffmpeg', 
        '-i', in_file1, 
        '-i', in_file2,
        '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]',
        '-map', '[vid]',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        out_file
    ])


def cat_two_videos_vertical(in_file1, in_file2, out_file):

    subprocess.call([
        'ffmpeg', 
        '-i', in_file1, 
        '-i', in_file2,
        '-filter_complex', '[0:v]pad=iw:ih*2[int];[int][1:v]overlay=0:H/2[vid]',
        '-map', '[vid]',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        out_file
    ])
