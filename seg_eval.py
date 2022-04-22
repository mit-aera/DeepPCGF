#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Test segmentation model and save results
"""

import os
import numpy as np
#import open3d
import torch.utils.data
from tqdm import tqdm
from PIL import Image
# project modules
from data.make_dataloader import make_dataloader
from options.test_options import TestOptions
from models import create_model
import utils.eval_utils as eval_utils

import torchvision.transforms as transforms
transform = transforms.Resize(size=(768, 1024), interpolation=2)

args = TestOptions().parse()
model = create_model(args) 
model.setup(args) 

if args.select_obj is not None:
    select_obj = [int(item) for item in args.select_obj.split(',')]
else:
    select_obj=None
test_loader = make_dataloader(args.dataset, args.data_path, 'test',
    1, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=False, select_obj=select_obj, 
    #do_augmentation=args.do_augmentation,
    image_based=args.image_based)
test_set = test_loader.dataset
      
def save_prediction(path_to_gt_mask, pred_seg):
    folder, obj_name, file_name = path_to_gt_mask.split('/')[-3:]
    test_output_dir = os.path.join(
        args.checkpoints_dir, args.name, 'images_test_768_1024_bilinear', folder, obj_name)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    
    pred_seg.save(os.path.join(test_output_dir, file_name))

def test():
    print('running segmentation model, evaluating results...')
    assert args.batch_size == 1, "batch size can only be 1"
    num_classes = test_set.get_num_of_models()
    iou_res = eval_utils.IoU(num_classes)
    for i, batch in enumerate(tqdm(test_loader)):
        if i > 1000:
            break
        label_pred = np.zeros((args.image_height, args.image_width))
        with torch.no_grad():
            model.set_input(batch) 
            model_index = model.model_index
            model_id = test_set.obj_ids[model_index]

            res = model.forward()  
            res = res.cpu().numpy()
            label_pred[res[0, :] == 1] = 255
            
            iou_res.add(res[0, :], batch['gt_mask'].cpu().numpy()[0], model_index)
            if args.save_seg:
                img = Image.fromarray(label_pred.astype(np.uint8))
                # img = transform(img)
                save_prediction(test_set.paths['gt_mask'][i], img)
                
    for i in range(num_classes):
        if iou_res.res[i, 2] == 0:
            continue
        model_id = test_set.obj_ids[i]
        model_name = test_set.obj_dics[model_id]
        print('Model {0} avg. IoU: {1}'.format(
            model_name, iou_res.res[i, 2]))
    res_all = np.sum(iou_res.res, 0)
    print('Overal IoU: {0}'.format(res_all[0] / res_all[1]))
    
if __name__ == '__main__':
    test()
