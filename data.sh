python /workspace/bisenet/BiSeNet/tools/data_process.py --imgs_path /workspace/data/855G2/images/train \
                                                        --save_path /workspace/data/855G2/labels/train \
                                                        --json_file /workspace/data/855G2/images/train/change_labels_new.json
python /workspace/bisenet/BiSeNet/tools/gen_dataset_annos.py --dataset coco \
                                                             --root_path /workspace/data/855G2 \
                                                          --save_path /workspace/data/855G2 \
                                                             --modes train