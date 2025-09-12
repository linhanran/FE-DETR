#!/bin/bash
source ~/.bashrc
source activate deim
CUDA_VISIBLE_DEVICES=0 python train.py -c /public/home/linhanran2023/FE-DETR-main/configs/FE-DETR/FE-DETR.yml --seed=0
