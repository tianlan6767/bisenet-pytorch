import os.path as osp
import os
import json
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import argparse
import imagesize


def make_dir(dst):
    if not osp.exists(dst):
        os.makedirs(dst, exist_ok=True)
    return dst

def get_jf(jf):
    with open(jf, "r") as f:
        json_data = json.load(f)
    return json_data


def creat_masks(h, w, seg_yxs, class_ids):
    mask_bak = np.zeros((h, w), dtype=np.uint8)
    for seg_yx, class_id in zip(seg_yxs, class_ids):
        cv2.fillPoly(mask_bak, [np.squeeze(seg_yx)], color=int(class_id))
    return mask_bak


def json2mask(src, dst, jf):
    jsd = get_jf(jf)
    imps = glob(src + "/*.[jb][pm][gp]")
    make_dir(dst)
    classes_ids = []
    for imp in tqdm(imps):
        imn = osp.basename(imp)
        w, h = imagesize.get(imp)
        try:
            regions = jsd[imn]["regions"]
        except:
            regions = []
        class_ids = []

        seg_yxs = []
        for region in regions:
            class_id = region["region_attributes"]["regions"]
            xs = region["shape_attributes"]["all_points_x"]
            ys = region["shape_attributes"]["all_points_y"]
            seg_yx = np.dstack((xs, ys))
            class_ids.append(class_id)
            seg_yxs.append(seg_yx)
        classes_ids +=class_ids
        mark = creat_masks(h, w, seg_yxs, class_ids)
        if not osp.exists(osp.join(dst, imn.replace(".bmp", ".png"))):
            cv2.imwrite(osp.join(dst, imn.replace(".bmp", ".png")), mark)
    print(sorted(tuple(set(classes_ids))))



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--imgs_path', type=str, default='',)
    parse.add_argument('--save_path', type=str, default='',)
    parse.add_argument('--json_file', type=str, default='',)
    args = parse.parse_args()

    json2mask(args.imgs_path, args.save_path, args.json_file)