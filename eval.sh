#!/bin/bash

# model="DiT-S/2"
# ckpt_path="/mnt/dit_moe_s_8E2A.pt"

model="DiT-XL/2"
ckpt_path="/mnt/dit_moe_xl_8E2A.pt"

vae_path="/mnt/vae"
num_experts="8"
num_sample_steps="1000"
image_size="256"
cfg_scale="1.5"
num_fid_samples="5000"
sample_dir="/root/autodl-tmp"

CUDA_VISIBLE_DEVICES=0

# sample generation
# python3 sample.py \
# --model $model \
# --ckpt $ckpt_path \
# --sample-dir $sample_dir \
# --vae-path $vae_path \
# --image-size $image_size \
# --cfg-scale $cfg_scale \
# --num_experts $num_experts \
# --num-sampling-steps $num_sample_steps \
# --num-fid-samples $num_fid_samples \
# --rf \

# create npz for evaluation
python3 evaluation/create_npz.py \
--num_samples $num_fid_samples \

# evaluate
ref_batch="/mnt/VIRTUAL_imagenet256_labeled.npz"
gen_batch="/root/autodl-tmp/DiT-XL-2-dit_moe_xl_8E2A-size-256-cfg-1.5-seed-2024-singlegpu.npz"
python3 evaluation/evaluation.py $ref_batch $gen_batch \
