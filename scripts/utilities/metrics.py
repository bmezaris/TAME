import math

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import metrics


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    mask = mask.cpu().data
    mask = mask.numpy()
    mask = mask[0, 0, :, :]

    img = img.cpu().data.numpy()
    img = img[0, :, :, :]
    img = np.transpose(img, (1, 2, 0))

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def normalizeWithMax(Att_map):
    x1_max = torch.max(Att_map, dim=3, keepdim=True)[0].max(2, keepdim=True)[0].detach()
    Att_map = (Att_map) / (x1_max)  # values now in [0,1]
    return Att_map


def normalizeMinMax4Dtensor(Att_map):
    x1_min = torch.min(Att_map, dim=3, keepdim=True)[0].min(2, keepdim=True)[0].detach()
    x1_max = torch.max(Att_map, dim=3, keepdim=True)[0].max(2, keepdim=True)[0].detach()
    Att_map = (Att_map - x1_min) / (x1_max - x1_min)  # values now in [0,1]
    return Att_map


def normalizeMinMax(cam_map):
    cam_map_min, cam_map_max = cam_map.min(), cam_map.max()
    cam_map -= torch.min(cam_map)
    cam_map /= (torch.max(cam_map) - torch.min(cam_map))
    return cam_map


def drop_Npercent(cam_map, percent):
    # Select N percent of pixels
    if percent == 0:
        return cam_map

    N, C, H, W = cam_map.size()
    cam_map_tmp = cam_map
    f = torch.flatten(cam_map)
    value = int(H * W * percent)
    # print(value)
    m = torch.kthvalue(f, value)
    cam_map_tmp[cam_map_tmp < m.values] = 0
    num_pixels = math.ceil((1 - percent) * (H * W))
    k = torch.count_nonzero(cam_map_tmp > 0) - num_pixels
    k = math.floor(k)
    if k >= 1:
        indices = torch.nonzero(cam_map == m.values)
        for pi in range(0, int(k)):
            cam_map_tmp[indices[pi][0], indices[pi][1], indices[pi][2], indices[pi][3]] = 0
    cam_map = cam_map_tmp
    #   cam_map[cam_map!=0] = 1
    # k = torch.count_nonzero(cam_map_tmp>0) - num_pixels
    #    print(k)
    return cam_map


def normalize(tensor):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
    return normalize


def accuracy(logits, target, topk=(1,)):
    '''
    Compute the top k accuracy of classification results.
    :param target: the ground truth label
    :param topk: tuple or list of the expected k values.
    :return: A list of the accuracy values. The list has the same lenght with para: topk
    '''
    maxk = max(topk)
    batch_size = target.size(0)
    scores = logits
    _, pred = scores.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_AUC(gt_labels, pred_scores):
    res = metrics.roc_auc_score(gt_labels, pred_scores)
    return res


def _to_numpy(v):
    v = torch.squeeze(v)
    if torch.is_tensor(v):
        v = v.cpu()
        v = v.numpy()
    elif isinstance(v, torch.autograd.Variable):
        v = v.cpu().data.numpy()
    return v


def AD(Yc_realImage, Yc_E):
    L = (sum(np.divide((Yc_realImage - Yc_E).clip(min=0)
                       , Yc_realImage))
         * (100 / len(Yc_realImage)))
    # print('AD', L)
    return L


def IC(Yc_realImage, Yc_E):
    dif = Yc_E - Yc_realImage
    sum_ = (sum(i > 0 for i in dif))
    count = sum_ * (100.0 / len(Yc_realImage))
    #  print('IC',count)
    return count
