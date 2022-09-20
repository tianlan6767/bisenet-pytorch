
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=2,
    num_aux_heads=4,
    lr_start=1e-3,
    weight_decay=1e-4,
    warmup_iters=100,
    max_iter=4000,
    dataset='CocoStuff',
    im_root='/workspace/data/855G3',
    train_im_anns='/workspace/data/855G3/train.txt',
    val_im_anns='/workspace/data/855G3/train.txt',
    scales=[0.75, 2.],
    cropsize=[2048, 2048],
    eval_crop=[2048, 2048],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=4,
    eval_ims_per_gpu=4,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res-ly',
)
