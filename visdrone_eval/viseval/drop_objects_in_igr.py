import numpy as np


def create_int_img(img):
    int_img = np.cumsum(img, axis=0)
    np.cumsum(int_img, axis=1, out=int_img)
    return int_img


def drop_objects_in_igr(gt, det, img_height, img_width):
    gt_ignore_mask = gt[:, 5] == 0
    curgt = gt[np.logical_not(gt_ignore_mask)]
    igr_region = gt[gt_ignore_mask, :4].clip(min=1)
    if len(igr_region):
        igr_map = np.zeros((img_height, img_width), dtype=np.int)

        for igr in igr_region:
            x1 = igr[0]
            y1 = igr[1]
            x2 = min(x1 + igr[2], img_width)
            y2 = min(y1 + igr[3], img_height)
            igr_map[y1 - 1:y2, x1 - 1:x2] = 1
        int_igr_map = create_int_img(igr_map)
        idx_left_gt = []

        for i, gtbox in enumerate(curgt):
            pos = np.round(gtbox[:4]).astype(np.int32).clip(min=1)
            x = max(1, min(img_width - 1, pos[0]))
            y = max(1, min(img_height - 1, pos[1]))
            w = pos[2]
            h = pos[3]
            tl = int_igr_map[y - 1, x - 1]
            tr = int_igr_map[y - 1, min(img_width, x + w) - 1]
            bl = int_igr_map[max(1, min(img_height, y + h)) - 1, x - 1]
            br = int_igr_map[max(1, min(img_height, y + h)) - 1,
                             min(img_width, x + w) - 1]
            igr_val = tl + br - tr - bl
            if igr_val / (h * w) < 0.5:
                idx_left_gt.append(i)

        curgt = curgt[idx_left_gt]

        idx_left_det = []
        for i, dtbox in enumerate(det):
            pos = np.round(dtbox[:4]).astype(np.int32).clip(min=1)
            x = max(1, min(img_width - 1, pos[0]))
            y = max(1, min(img_height - 1, pos[1]))
            w = pos[2]
            h = pos[3]
            tl = int_igr_map[y - 1, x - 1]
            tr = int_igr_map[y - 1, min(img_width, x + w) - 1]
            bl = int_igr_map[max(1, min(img_height, y + h)) - 1, x - 1]
            br = int_igr_map[max(1, min(img_height, y + h)) - 1,
                             min(img_width, x + w) - 1]
            igr_val = tl + br - tr - bl
            if igr_val / (h * w) < 0.5:
                idx_left_det.append(i)

        det = det[idx_left_det]

    return curgt, det
