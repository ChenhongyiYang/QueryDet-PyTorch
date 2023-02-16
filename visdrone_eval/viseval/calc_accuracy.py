import numpy as np
from .bbox_overlaps import bbox_overlaps


def eval_res(gt0, dt0, thr):
    """
    :param gt0: np.array[ng, 5], ground truth results [x, y, w, h, ignore]
    :param dt0: np.array[nd, 5], detection results [x, y, w, h, score]
    :param thr: float, IoU threshold
    :return gt1: np.array[ng, 5], gt match types
             dt1: np.array[nd, 6], dt match types
    """
    nd = len(dt0)
    ng = len(gt0)

    # sort
    dt = dt0[dt0[:, 4].argsort()[::-1]]
    gt_ignore_mask = gt0[:, 4] == 1
    gt = gt0[np.logical_not(gt_ignore_mask)]
    ig = gt0[gt_ignore_mask]
    ig[:, 4] = -ig[:, 4]  # -1 indicates ignore

    dt_format = dt[:, :4].copy()
    gt_format = gt[:, :4].copy()
    ig_format = ig[:, :4].copy()
    dt_format[:, 2:] += dt_format[:, :2]  # [x2, y2] = [w, h] + [x1, y1]
    gt_format[:, 2:] += gt_format[:, :2]
    ig_format[:, 2:] += ig_format[:, :2]

    iou_dtgt = bbox_overlaps(dt_format, gt_format, mode='iou')
    iof_dtig = bbox_overlaps(dt_format, gt_format, mode='iof')
    oa = np.concatenate((iou_dtgt, iof_dtig), axis=1)

    # [nd, 6]
    dt1 = np.concatenate((dt, np.zeros((nd, 1), dtype=dt.dtype)), axis=1)
    # [ng, 5]
    gt1 = np.concatenate((gt, ig), axis=0)

    for d in range(nd):
        bst_oa = thr
        bstg = -1  # index of matched gt
        bstm = 0  # best match type
        for g in range(ng):
            m = gt1[g, 4]
            # if gt already matched, continue to next gt
            if m == 1:
                continue
            # if dt already matched, and on ignore gt, nothing more to do
            if bstm != 0 and m == -1:
                break
            # continue to next gt until better match is found
            if oa[d, g] < bst_oa:
                continue
            bst_oa = oa[d, g]
            bstg = g
            bstm = 1 if m == 0 else -1  # 1: matched to gt, -1: matched to ignore

        # store match type for dt
        dt1[d, 5] = bstm
        # store match flag for gt
        if bstm == 1:
            gt1[bstg, 4] = 1

    return gt1, dt1


def voc_ap(rec, prec):
    mrec = np.concatenate(([0], rec, [1]))
    mpre = np.concatenate(([0], prec, [0]))
    for i in reversed(range(0, len(mpre)-1)):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap


def calc_accuracy(num_imgs, all_gt, all_det, per_class=False):
    """
    :param num_imgs: int
    :param all_gt: list of np.array[m, 8], [:, 4] == 1 indicates ignored regions,
                    which should be dropped before calling this function
    :param all_det: list of np.array[m, 6], truncation and occlusion not necessary
    :param per_class:
    """
    assert num_imgs == len(all_gt) == len(all_det)

    ap = np.zeros((10, 10), dtype=np.float32)
    ar = np.zeros((10, 10, 4), dtype=np.float32)
    eval_class = []

    print('')
    for id_class in range(1, 11):
        print('evaluating object category {}/10...'.format(id_class))

        for gt in all_gt:
            if np.any(gt[:, 5] == id_class):
                eval_class.append(id_class - 1)

        x = 0
        for thr in np.linspace(0.5, 0.95, num=10):
            y = 0
            for max_dets in (1, 10, 100, 500):
                gt_match = []
                det_match = []
                for gt, det in zip(all_gt, all_det):
                    det_limited = det[:min(len(det), max_dets)]
                    mask_gt_cur_class = gt[:, 5] == id_class
                    mask_det_cur_class = det_limited[:, 5] == id_class
                    gt0 = gt[mask_gt_cur_class, :5]
                    dt0 = det_limited[mask_det_cur_class, :5]
                    gt1, dt1 = eval_res(gt0, dt0, thr)
                    # 1: matched, 0: unmatched, -1: ignore
                    gt_match.append(gt1[:, 4])
                    # [score, match type]
                    # 1: matched to gt, 0: unmatched, -1: matched to ignore
                    det_match.append(dt1[:, 4:6])
                gt_match = np.concatenate(gt_match, axis=0)
                det_match = np.concatenate(det_match, axis=0)

                idrank = det_match[:, 0].argsort()[::-1]
                tp = np.cumsum(det_match[idrank, 1] == 1)
                rec = tp / max(1, len(gt_match))  # including ignore (already dropped)
                if len(rec):
                    ar[id_class - 1, x, y] = np.max(rec) * 100

                y += 1

            fp = np.cumsum(det_match[idrank, 1] == 0)
            prec = tp / (fp + tp).clip(min=1)
            ap[id_class - 1, x] = voc_ap(rec, prec) * 100

            x += 1

    ap_all = np.mean(ap[eval_class, :])
    ap_50 = np.mean(ap[eval_class, 0])
    ap_75 = np.mean(ap[eval_class, 5])
    ar_1 = np.mean(ar[eval_class, :, 0])
    ar_10 = np.mean(ar[eval_class, :, 1])
    ar_100 = np.mean(ar[eval_class, :, 2])
    ar_500 = np.mean(ar[eval_class, :, 3])

    results = (ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500)

    if per_class:
        ap_classwise = np.mean(ap, axis=1)
        results += (ap_classwise,)

    print('Evaluation completed. The performance of the detector is presented as follows.')

    return results
