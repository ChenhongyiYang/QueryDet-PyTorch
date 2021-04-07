import torch
import numpy as np
import cv2

from detectron2.structures import Boxes


def get_box_scales(boxes: Boxes):
    return torch.sqrt((boxes.tensor[:, 2] - boxes.tensor[:, 0]) * (boxes.tensor[:, 3] - boxes.tensor[:, 1]))

def get_anchor_center_min_dis(box_centers: torch.Tensor, anchor_centers: torch.Tensor):
    """
    Args:
        box_centers: [N, 2]
        anchor_centers: [M, 2]
    Returns:
        
    """
    N, _ = box_centers.size()
    M, _ = anchor_centers.size()
    if N == 0:
        return torch.ones_like(anchor_centers)[:, 0] * 99999, (torch.zeros_like(anchor_centers)[:, 0]).long()
    acenters = anchor_centers.view(-1, 1, 2)
    acenters = acenters.repeat(1, N, 1)
    bcenters = box_centers.view(1, -1, 2)
    bcenters = bcenters.repeat(M, 1, 1)
    
    dis = torch.sqrt(torch.sum((acenters - bcenters)**2, dim=2))

    mindis, minind = torch.min(input=dis, dim=1)

    return mindis, minind
           












