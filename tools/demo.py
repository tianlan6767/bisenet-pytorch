
import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file
from glob import glob 
from pathlib import Path
import os.path as osp
import time, json
import os
os.environ['CUDA_VISBLE_DEVICES'] ='0, 1, 2'

# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)


def save_json(dir, jsd, name):
    if osp.exists(osp.join(dir, "{}.json".format(name))):
        cur_time = time.strftime("%m%d%H%M", time.localtime(time.time()))
        name = name + "_" + cur_time
    with open(osp.join(dir, "{}.json".format(name)), "w", encoding="utf-8") as f:
        json.dump(jsd, f, ensure_ascii=False)



def mask2json(img_seg, imn, label_count=25, min_region=50):
    """
        将掩模图转换为标注信息，掩模图的灰度值对应缺陷标签

    :param image_folder: 掩模图（*.bmp）文件夹
    :param json_image_ext: 标注文件图像名称后缀
    :param label_count: 缺陷标签总数
    :param min_region: 缺陷最小面积，小于此阈值的缺陷将忽略
    :return: None
    """

    image = np.asarray(img_seg, dtype=np.uint8)
    if cv2.countNonZero(image) == 0:
        return {"filename":imn, "regions":[], "type": "inf"}
    content = []
    for label in range(1, label_count + 1):
        src_image = image.copy()
        ret, binary = cv2.threshold(src_image, label, 255, cv2.THRESH_TOZERO_INV)
        ret, bin_image = cv2.threshold(binary, label - 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            region_area = cv2.contourArea(contour)
            if region_area >= min_region:
                region = dict(region_attributes={'regions': str(label), "score": 0.6})
                new_contour = contour.squeeze()
                region['shape_attributes'] = {
                    'all_points_x': [int(pt[0]) for pt in new_contour],
                    'all_points_y': [int(pt[1]) for pt in new_contour]
                }
                content.append(region)

    if len(content):
        # img_info = {imn:{"filename":imn, "regions":content, "type": "inf"}}
        img_info = {"filename":imn, "regions":content, "type": "inf"}
    else:
        # img_info = {imn:{"filename":imn, "regions":[], "type": "inf"}}
        img_info = {"filename":imn, "regions":[], "type": "inf"}

    return img_info

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='/workspace/bisenet/BiSeNet/configs/bisenetv2_coco.py',)
parse.add_argument('--weight-path', type=str, default='/workspace/bisenet/BiSeNet/res1/model_4999.pth',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)
src = r'/workspace/data/855G2/images/train'
dst = r"/workspace/data/855G2/test"
Path(dst).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda:2")
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0] = np.zeros((3), dtype=np.uint8)
# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
# net.cuda()
net.to(device)

to_tensor = T.ToTensor(
    mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
    std=(0.27469736, 0.27012361, 0.28515933),
)

imps = glob(src + "/*.bmp")
imgs_info = {}
inf_times = []
for imp in tqdm(imps):
    imn = osp.basename(imp)
    im = cv2.imread(imp,-1)[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).to(device)

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    st = time.time()
    # print("推理圖片--------------", imn)
    out = net(im)[0]

    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    out = out.argmax(dim=1)
    et = round(time.time() - st, 5)
    inf_times.append(et)
    # visualize    
    out = out.squeeze().detach().cpu().numpy()

    info = mask2json(out, imn, label_count=25, min_region=2)
    # imgs_info.update(info)
    # print("保存json------------------------",imn)
    imgs_info[imn] = info

    pred = palette[out]
    # cv2.imwrite(osp.join(dst, imn.replace(".bmp", ".png")), pred)
save_json(dst, imgs_info, "info_bisenet")
print("{}张图片,每张图片耗时{}".format(len(inf_times), sum(inf_times)/len(inf_times)))
print(inf_times)

