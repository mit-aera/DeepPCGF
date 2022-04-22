#!/bin/bash
lognrun(){ echo "$@"; "$@"; }
#lognrun some_command -o "some_option" > logfile


# trian the model
data_id='0'
geometric_check='vanilla'
gpu_ids='0'
mu='1000'
load_iter='65'
w='1024'
h='768'
select_pts='70'

exp_name=('blackbird') 
dataset=('BlackBird')
data_path=('../data/Blackbird')

#lognrun python train.py --name ${exp_name[${data_id}]} --model gcn --image_based true --dataset ${dataset[${data_id}]} --data_path ${data_path[${data_id}]} --voxel_size 0.02  --geometric_check ${geometric_check} --gpu_ids ${gpu_ids}  --image_width ${w} --image_height ${h} --mu ${mu} --continue_train --load_iter 0 >> ../checkpoints/${exp_name[${data_id}]}/train.out

#lognrun python train.py --name ${exp_name[${data_id}]} --model gcn --image_based true --dataset ${dataset[${data_id}]} --data_path ${data_path[${data_id}]} --voxel_size 0.02  --geometric_check ${geometric_check} --gpu_ids ${gpu_ids}  --image_width ${w} --image_height ${h} --mu ${mu} >> ../checkpoints/${exp_name[${data_id}]}/train.out

lognrun python test.py \
  --name ${exp_name[${data_id}]} \
  --model gcn \
  --image_based true \
  --dataset ${dataset[${data_id}]} \
  --data_path ${data_path[${data_id}]} \
  --voxel_size 0.02  \
  --geometric_check ${geometric_check} \
  --gpu_ids ${gpu_ids} \
  --image_width ${w} \
  --image_height ${h} \
  --load_iter ${load_iter} \
  --select_pts ${select_pts} 
