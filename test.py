import time
import numpy as np
from tqdm import tqdm
import torch.utils.data

import utils.open3d_utils as o3d_utils
from models import create_model
from utils.custom_logger import Logger
import utils.eval_utils as eval_utils
import data.tf_numpy as tf_numpy
from utils.metrics import AverageMeter, res_summary
from data.make_dataloader import make_dataloader
from options.test_options import TestOptions



args = TestOptions().parse()
logger = Logger(args)
model = create_model(args) 
model.setup(args) 

test_loader = make_dataloader(args.dataset, args.data_path, 'test',
    1, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=False, select_obj=args.select_obj, do_augmentation=False,
    image_based=args.image_based)
   
test_set = test_loader.dataset
model.set_dataset(test_set)
  
def test():
    print()
    print('Running %s model...' % args.model)
    assert args.batch_size == 1, "batch size can only be 1"
    num_models = test_set.get_num_of_models()
    total_instance_cnt = [0 for i in range(num_models)]
    success_cnt = [0 for i in range(num_models)]
    
    rte_meter = AverageMeter('RTE')
    rre_meter = AverageMeter('RRE')
    t_x_meter = AverageMeter('RTE_x')
    t_y_meter = AverageMeter('RTE_y')
    t_z_meter = AverageMeter('RTE_z')
    adds_meter = AverageMeter('ADDS')

    start = time.time()
    dataloader_timer = AverageMeter('data_loading_time') 
    runtime_timer = AverageMeter('overal_time')
    astart = time.time()

    pose_preds = np.zeros((len(test_loader), 4, 4), dtype=np.float32)
    pose_preds_refine = np.zeros((len(test_loader), 4, 4), dtype=np.float32)
    for i, batch in enumerate(tqdm(test_loader)):
        dataloader_timer.update(time.time()-start)
        start = time.time()
        model_index = batch['model'][0].get_index()
        model_id = test_set.obj_ids[model_index]
        pose_gt = batch['T_gt'][0, :].numpy()
        diameter = test_set.models[model_id].get_model_diameter()
        obj = batch['model'][0] 
        model_points = obj.get_model_points()
        with torch.no_grad():
            model.set_input(batch)
            pose_pred = model.forward()  
            rot_pred = pose_pred[:3, :3]
            t_pred = pose_pred[:3, 3]

            quat_pred = tf_numpy.quaternion_from_matrix(rot_pred)
            t_pred = t_pred[:, None]
            pose_preds[i, :, :] = pose_pred

            if args.do_icp_refine:
                rot_refined, t_refined = o3d_utils.icp_refine(rot_pred, t_pred, 
                    model_points, 
                    model.xyz_s.cpu().numpy())
                quat_refined = tf_numpy.quaternion_from_matrix(rot_refined)

                rot_pred = rot_refined
                t_pred = t_refined
                quat_pred = quat_refined
                
                pose_pred_refine = np.zeros((4, 4))
                pose_pred_refine[:3, :3] = rot_pred
                pose_pred_refine[:3, 3] = t_pred[:, 0]
                pose_preds_refine[i, :, :] = pose_pred_refine

            # Relative Translation Error (RTE)
            rte = eval_utils.rte(t_pred, pose_gt[:3, 3][:, None])
            rte_meter.update(rte)
            # Relative Rotation Error (RRE)
            rre = eval_utils.rre(rot_pred, pose_gt[:3, :3])
            if not np.isnan(rre):
                rre_meter.update(rre)
            # Ralative Translation Error in x, y, z direction
            q_distance, quat_angle, x_offset, y_offset, z_offset = \
                eval_utils.compute_error(quat_pred, t_pred, 
                    tf_numpy.quaternion_from_matrix(pose_gt[:3, :3]),\
                    pose_gt[:3, 3][:, None])
            t_x_meter.update(x_offset)
            t_y_meter.update(y_offset)
            t_z_meter.update(z_offset)

            # Average closest point distance ( ADD(S) )
            distance = eval_utils.compute_adds_metric(
                rot_pred, t_pred, 
                pose_gt[:3, :3], pose_gt[:3, 3][:, None],
                model_points,
                test_set.is_symmetric_obj(model_index))
            adds_meter.update(distance)
            threshold = 0.1
            if eval_utils.is_correct_pred(distance, diameter, threshold):
                success_cnt[model_index] += 1
            total_instance_cnt[model_index] += 1
            end = time.time()
            runtime_timer.update(end-start)
            start = end

    # result summary
    print()
    print('time elapse:', time.time() - astart)
    timing = res_summary([dataloader_timer, model.seg_timer, model.forward_timer, model.gc_timer,
        runtime_timer])
    logger.print_current_statistics(timing, is_train=False)
    res = res_summary([rte_meter, t_x_meter, t_y_meter,
        t_z_meter, rre_meter, adds_meter])
    logger.print_current_statistics(res, is_train=False)
    logger.print_success_rate(num_models, total_instance_cnt, 
        success_cnt, test_set.obj_ids, test_set.obj_dics)

def main():
    test()
    
if __name__ == '__main__':
    main()
