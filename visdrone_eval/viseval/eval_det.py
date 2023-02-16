from .calc_accuracy import calc_accuracy
from .drop_objects_in_igr import drop_objects_in_igr


def eval_det(all_gt, all_det, allheight, allwidth, per_class=False):
    """
    :param all_gt: list of np.array[m, 8]
    :param all_det: list of np.array[m, 6], truncation and occlusion not necessary
    :param allheight:
    :param allwidth:
    :param per_class:
    """
    all_gt_ = []
    all_det_ = []
    num_imgs = len(all_gt)
    for gt, det, height, width in zip(all_gt, all_det, allheight, allwidth):
        gt, det = drop_objects_in_igr(gt, det, height, width)
        gt[:, 4] = 1 - gt[:, 4]  # set ignore flag
        all_gt_.append(gt)
        all_det_.append(det)
    return calc_accuracy(num_imgs, all_gt_, all_det_, per_class)
