#!/bin/bash
lognrun(){ echo "$@"; "$@"; }

data_id='0'
geometric_check='gcn'
gpu_ids='0'
mu='100'
load_iter='120'
select_pts='70'
exp_name=('LineMOD' 'OcclusionLineMOD')
checkpoints_dir='../checkpoints'
dataset=('LineMOD' 'LineMODOcclusion')
data_path=('../data/Linemod_preprocessed' '../data/Linemod_occlusion')

is_train=false
#is_train=true
if $is_train; then
# trian the model
lognrun python train.py \
  --name ${exp_name[${data_id}]} \
  --model pcgf \
  --checkpoints_dir ${checkpoints_dir} \
  --image_based true \
  --dataset ${dataset[${data_id}]} \
  --data_path ${data_path[${data_id}]} \
  --voxel_size 0.003  \
  --geometric_check ${geometric_check} \
  --gpu_ids ${gpu_ids}  \
  --mu ${mu} \
#  --continue_train
#  --load_iter ${load_iter} 
else
# model evaluation 
lognrun python test.py --name ${exp_name[${data_id}]} \
  --model pcgf \
  --checkpoints_dir ${checkpoints_dir} \
  --image_based true \
  --dataset ${dataset[${data_id}]} \
  --data_path ${data_path[${data_id}]} \
  --voxel_size 0.003  \
  --geometric_check ${geometric_check} \
  --gpu_ids ${gpu_ids} \
  --select_pts ${select_pts} 
  #--load_iter ${load_iter} \
fi
