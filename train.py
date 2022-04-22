import time
import os
import numpy as np
import torch.utils.data
from tqdm import tqdm

from models import create_model
from options.train_options import TrainOptions
from utils.custom_logger import Logger
from data.make_dataloader import make_dataloader
from utils.metrics import AverageMeter, res_summary
from utils.eval_utils import compute_adds_metric, is_correct_pred
import utils.eval_utils as eval_utils

import matplotlib.pyplot as plt

args = TrainOptions().parse()
logger = Logger(args)
model = create_model(args) 
model.setup(args) 

train_loader = make_dataloader(args.dataset, args.data_path, args.phase,
    args.batch_size, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=True, select_obj=args.select_obj, do_augmentation=args.do_augmentation, 
    image_based=args.image_based)
val_loader = make_dataloader(args.dataset, args.data_path, 'test',
    1, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=False, select_obj=args.select_obj, do_augmentation=args.do_augmentation,
    image_based=args.image_based)
train_set = train_loader.dataset
val_set = val_loader.dataset

model.set_dataset(train_set)

def train(epoch):
    model.set_phase('train')
    
    iter_data_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if not batch:
            continue 
        iter_start_time = time.time()
        if i % args.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        model.set_input(batch)       
        if (i+1) % args.step_freq == 0 and i != 0:
            step = True
        else:
            step = False
        if i > 300:
            model.alpha = 1
            model.mu = args.mu
        model.optimize_parameters(step)

        with torch.no_grad():
            if i % args.print_freq == 0:    
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / args.batch_size
                logger.print_current_losses(
                    epoch, i, losses, t_comp, len(train_loader))

            #if args.model == 'seg' and i % args.display_freq == 0:   
            #    rgb_images = [batch['rgb']]
                #print(batch)
                #mask_images = [batch['gt_mask'], model.seg_out]
                # print(model.seg_out.shape)
                # print(gt_seg.shape, pred_seg.shape, coord.shape)
                #depth_images = []#[batch['seg_mask'], model.out]
                #logger.vis_images(
                #     'train', epoch, i, len(train_loader), rgb_images, mask_images)
            # if i % args.display_freq == 0:   
            #     rgb_images = [batch['rgb'][0]]
            #     gt_seg = batch['seg_mask'].squeeze()
            #     pred_seg = model.seg_out
                
            #     coord = train_loader.dataset.cam2img(batch['xyz_d'].numpy())
            #     coord = torch.from_numpy(coord).long()
                
            #     # print(gt_seg.shape, pred_seg.shape, coord.shape)
            #     depth_images = []#[batch['seg_mask'], model.out]
            #     logger.save_and_display_images(
            #         'train', epoch, i, len(train_loader), rgb_images, gt_seg, pred_seg, coord)

            iter_data_time = time.time()
            # t_run = iter_data_time - iter_start_time


    model.update_learning_rate()

# only use in debug mode
def validate(epoch):
    model.set_phase('val')
    len_loader = len(val_loader)
    loss_meter, rte_meter, rre_meter, adds_meter = AverageMeter('L1Loss'), \
        AverageMeter('RTE'), AverageMeter('RRE'), AverageMeter('ADDS')

    num_models = val_set.get_num_of_models()
    total_instance_cnt = [0 for i in range(num_models)]
    success_cnt = [0 for i in range(num_models)]
    for i, batch in tqdm(enumerate(val_loader)):
        #if i > 100:
        #    break
        if i % 20 != 0:
            continue

        with torch.no_grad():
            model.set_input(batch)  
            T_est = model.forward() 
            T_gt = batch['T_gt'][0].numpy()

            rte = eval_utils.rte(T_est[:3, 3], T_gt[:3, 3][:, None])
            rre = eval_utils.rre(T_est[:3, :3], T_gt[:3, :3])
            rte_meter.update(rte)
            if not np.isnan(rre):
                rre_meter.update(rre)
            tl1loss = np.mean(np.abs(T_est[:3, 3] - T_gt[:3, 3]))
            loss_meter.update(tl1loss)

            obj = batch['model'][0] # currently val only supports batch size = 1 
            model_points = obj.get_model_points()
            diameter = obj.get_model_diameter()
            model_index = obj.get_index()
            is_sym = obj.is_symmetric()

            distance = compute_adds_metric(T_est[:3, :3], T_est[:3, 3][:, None], 
                T_gt[:3, :3], T_gt[:3, 3][:, None],
                model_points, is_sym) #currently val only supports batch size=1
            adds_meter.update(distance)
            if is_correct_pred(distance, diameter): 
                success_cnt[model_index] += 1
            total_instance_cnt[model_index] += 1

            torch.cuda.empty_cache()

            if i % 100 == 0 and i > 0:
                res = res_summary([loss_meter, rte_meter, rre_meter, adds_meter])
                logger.print_current_statistics(res, epoch, i, 
                    len_loader, is_train=False)

    res = res_summary([loss_meter, rte_meter, rre_meter, adds_meter])
    logger.print_current_statistics(res, epoch, len_loader, 
        len_loader, is_train=False)
    logger.print_success_rate(num_models, total_instance_cnt, 
        success_cnt, val_set.obj_ids, val_set.obj_dics)
    return adds_meter.avg

def seg_validate(epoch):
    print('running segmentation model, evaluating results...')
    num_classes = val_set.get_num_of_models()
    iou_res = eval_utils.IoU(num_classes)
    for i, batch in enumerate(tqdm(val_loader)):
        # if i > 100:
        #     break
        if i % 100 != 0:
            continue
        label_pred = np.zeros((args.image_height, args.image_width))
        with torch.no_grad():
            model.set_input(batch) 
            model_index = model.model_index
            model_id = val_set.obj_ids[model_index]

            res = model.forward() 
            res = res.cpu().numpy()
            label_pred[res[0, :] == 1] = 255
            
            iou_res.add(res[0, :], batch['gt_mask'].cpu().numpy()[0], model_index)
                
    for i in range(num_classes):
        if iou_res.res[i, 2] == 0:
            continue
        model_id = val_set.obj_ids[i]
        model_name = val_set.obj_dics[model_id]
        print('Model {0} avg. IoU: {1}'.format(
            model_name, iou_res.res[i, 2]))
    res_all = np.sum(iou_res.res, 0)
    iou = res_all[0] / res_all[1]
    print('Overal IoU: {0}'.format(iou))
    return iou
          
def main():
    best_dis = float("inf")
    best_iou = 0
    args.epoch_start = model.schedulers[0].state_dict()['last_epoch']
    for epoch in range(args.epoch_start, args.epoch_end):
        if epoch % args.valid_freq == 0 and epoch != 0:
            mean_dis = validate(epoch)
            if mean_dis < best_dis: 
                best_dis = mean_dis
                save_suffix = os.path.join(
                    args.checkpoints_dir, args.name, 'val_best')
                #model.save_networks(save_suffix)
                model.save_checkpoints(save_suffix)

        train(epoch)
        save_suffix = os.path.join(
           args.checkpoints_dir, args.name, 'latest')
        #model.save_networks(save_suffix)
        model.save_checkpoints(save_suffix)

if __name__ == '__main__':
    main()
