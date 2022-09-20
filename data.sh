python /workspace/bisenet/BiSeNet/tools/data_process.py --imgs_path /workspace/data/855G3/images/train \
                                                        --save_path /workspace/data/855G3/labels/train \
                                                        --json_file /workspace/data/855G3/images/train/format_via_export_json.json
python /workspace/bisenet/BiSeNet/tools/gen_dataset_annos.py --dataset coco \
                                                             --root_path /workspace/data/855G3 \
                                                          --save_path /workspace/data/855G3 \
                                                             --modes train