
import os
import os.path as osp
import argparse
import cv2
import numpy as np
from tqdm import tqdm



def gen_coco(root_path, save_path, models):
    '''
        root_path:
            |- images
                |- train2017
                |- val2017
            |- labels
                |- train2017
                |- val2017
    '''
    for mode in models:
        im_root = osp.join(root_path, f'images/{mode}')
        lb_root = osp.join(root_path, f'labels/{mode}')

        ims = [im for im in os.listdir(im_root) if im.endswith(".bmp")]
        lbs = [lb for lb in os.listdir(lb_root) if lb.endswith(".png")]
        assert len(ims) == len(lbs), "图片数量需要与标签数量一致"

        im_names = [el.replace('.bmp', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{mode}/{name}.bmp,labels/{mode}/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}/{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))
        print(f"save {mode}.txt successfully")
        if mode == "train":
            cal_mean_std(osp.join(save_path, f"{mode}.txt"))


def gen_ade20k():
    '''
        root_path:
            |- images
                |- training
                |- validation
            |- annotations
                |- training
                |- validation
    '''
    root_path = './datasets/ade20k/'
    save_path = './datasets/ade20k/'
    folder_map = {'train': 'training', 'val': 'validation'}
    for mode in ('train', 'val'):
        folder = folder_map[mode]
        im_root = osp.join(root_path, f'images/{folder}')
        lb_root = osp.join(root_path, f'annotations/{folder}')

        ims = [im for im in os.listdir(im_root) if im.endswith(".bmp")]
        lbs = [lb for lb in os.listdir(lb_root) if lb.endswith(".png")]

        print(len(ims))
        print(len(lbs))

        im_names = [el.replace('.jpg', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{folder}/{name}.jpg,annotations/{folder}/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}/{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))



def cal_mean_std(file_name):
    mode = osp.basename(file_name)[:-4]
    root_path = osp.dirname(file_name)
    with open(file_name,"r") as f:
        files = f.readlines()
        # print(files)
        files = [file.split(",")[0] for file in files]
    # print(files)

    MEAN = []
    STD = []

    for file in tqdm(files):
        img = cv2.imread(osp.join(root_path, file), -1)[:,:,::-1]/255.0
        MEAN.append(np.mean(img, axis=(0,1)))
        STD.append(np.std(img, axis=(0,1)))

    MEAN = np.array(MEAN)
    STD = np.array(STD)
    # print(MEAN.shape)
    # print(STD.shape)
    MEAN = np.mean(MEAN,axis=0)
    STD = np.mean(STD,axis=0)
    print(MEAN)
    print(STD)
    
    with open(osp.join(osp.dirname(file_name), f"{mode}_mean_std.txt"), "w") as fms:
        fms.write(f"mean={MEAN}")
        fms.write("\n")
        fms.write(f"std={STD}")
        print(f"save {mode}_mean_std.txt successfully")





if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='coco')
    parse.add_argument('--root_path', type=str, default='')
    parse.add_argument('--save_path', type=str, default='')
    parse.add_argument('--modes', action='append', nargs='*')
    args = parse.parse_args()
    if args.dataset == 'coco':
        gen_coco(args.root_path, args.save_path, args.modes[0])
    elif args.dataset == 'ade20k':
        gen_ade20k()
