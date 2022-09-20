# BiSeNetV1 & BiSeNetV2

## deploy trained models

1. tensorrt  
You can go to [tensorrt](./tensorrt) for details.  

2. ncnn  
You can go to [ncnn](./ncnn) for details.  

3. openvino  
You can go to [openvino](./openvino) for details.  

4. tis  
Triton Inference Server(TIS) provides a service solution of deployment. You can go to [tis](./tis) for details.


## platform

My platform is like this: 

* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.80.02
* cuda 10.2/11.3
* cudnn 8
* miniconda python 3.8.8
* pytorch 1.11.0


## get start

With a pretrained weight, you can run inference on an single image like this: 

```
$ python tools/demo.py --config configs/bisenetv2_city.py --weight-path /path/to/your/weights.pth --img-path ./example.png
```

This would run inference on the image and save the result image to `./res.jpg`.  

Or you can run inference on a video like this:  
```
$ python tools/demo_video.py --config configs/bisenetv2_coco.py --weight-path res/model_final.pth --input ./video.mp4 --output res.mp4
```
This would generate segmentation file as `res.mp4`. If you want to read from camera, you can set `--input camera_id` rather than `input ./video.mp4`.   


4.custom dataset 
```
    数据保存路径
        - root-path
            - images
               - train(*.bmp, *.json)
               - val()
               - test()
            -labels
               - train(*.txt)
    命令：[here](data.sh)
        修改：各文件对应的保存路径 
        *run data.sh**
```



## train

Training commands I used to train the models can be found in [here](./dist_train.sh).

Note:  
1. though `bisenetv2` has fewer flops, it requires much more training iterations. The the training time of `bisenetv1` is shorter.
2. I used overall batch size of 16 to train all models. Since cocostuff has 171 categories, it requires more memory to train models on it. I split the 16 images into more gpus than 2, as I do with cityscapes.


## finetune from trained model

You can also load the trained model weights and finetune from it, like this:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ torchrun --nproc_per_node=2 tools/train_amp.py --finetune-from ./res/model_final.pth --config ./configs/bisenetv2_city.py # or bisenetv1
```

## 数据修改：
* 配置参数修改 文件路径
    configs/bisenetv2_coco.py
    ```
        cfg = dict(
        model_type='bisenetv2',
        n_cats=2,     # 修改类别，包括背景
        num_aux_heads=4,
        lr_start=1e-3,  #
        weight_decay=1e-4,
        warmup_iters=100,
        max_iter=4000,
        dataset='CocoStuff',
        im_root='/workspace/data/855G3',
        train_im_anns='/workspace/data/855G3/train.txt',  # 训练集文件路径
        val_im_anns='/workspace/data/855G3/train.txt',    # 验证集文件路径
        scales=[0.75, 2.],
        cropsize=[2048, 2048],                         # 裁剪尺寸
        eval_crop=[2048, 2048],                        # 裁剪尺寸 
        eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        ims_per_gpu=4,
        eval_ims_per_gpu=4,
        use_fp16=True,
        use_sync_bn=True,
        respth='./res-ly',
)
    ```
* coco.py文件修改：文件路径：/workspace/bisenet/BiSeNet/lib/data/coco.py
    ```
    class CocoStuff(BaseDataset):

        def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
            super(CocoStuff, self).__init__(
                    dataroot, annpath, trans_func, mode)
            self.n_cats = 2 # 修改类别
            self.lb_ignore = 255

            ## label mapping, remove non-existing labels
            # missing = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]
            # remain = [ind for ind in range(182) if not ind in missing]
            self.lb_map = np.arange(2).astype(np.uint8)   # 修改数量
            # for ind in remain:
                # self.lb_map[ind] = remain.index(ind)

            self.to_tensor = T.ToTensor(
                mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
                std=(0.27469736, 0.27012361, 0.28515933),
            )   # 修改成计算好的结果
            # print("cocostuff---------------")

    ```
* 调整固定学习率：文件路径/workspace/bisenet/BiSeNet/lib/lr_scheduler.py
```
    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        # lrs = [0.00001 for _ in self.base_lrs]
        return lrs
```


## 模型训练[here](dist_train.sh)
* bash dist_train.sh

## 模型推理[here](./tools/demo.py)
* 修改配置文件路径，权重文件路径，图片路径
