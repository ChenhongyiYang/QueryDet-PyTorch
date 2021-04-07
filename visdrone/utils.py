import os
import cv2
import json



def read_label_txt(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()

    labels = []
    for line in lines:
        line = line.strip().split(',')

        x, y, w, h, not_ignore, cate, trun, occ = line[:8]

        labels.append(
            {'bbox': (int(x),int(y),int(w),int(h)), 
             'ignore': 0 if int(not_ignore) else 1, 
             'class': int(cate), 
             'truncate': int(trun),
             'occlusion': int(occ)}
        )
    return labels


def read_all_labels(ann_root):
    ann_list = os.listdir(ann_root)
    all_labels = {}
    for ann_file in ann_list:
        if not ann_file.endswith('txt'):
            continue
        ann_labels = read_label_txt(os.path.join(ann_root, ann_file))
        all_labels[ann_file] = ann_labels
    return all_labels









