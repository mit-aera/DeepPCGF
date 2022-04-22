import numpy as np
import torch
import os, sys
import datetime
from utils.logger_utils import *
import torchvision



if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def create_folder(opt):
    # current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    base_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(base_dir): 
        os.makedirs(base_dir)

    train_output_dir = os.path.join(base_dir, 'images_train')
    val_output_dir = os.path.join(base_dir, 'images_val')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    return base_dir, train_output_dir, val_output_dir

def NED2EDN(pts):
    tf = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])

    return np.matmul(pts, tf[:3, :3].T)

def NED2EDNPose(pose_ned):
    tf = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])
    return tf @ pose_ned

def draw_bbox(index, dset, im, obj, pose_gt, pose_pred, path):
    # pose_gt = NED2EDNPose(pose_gt)
    # pose_pred = NED2EDNPose(pose_pred)

    pts = obj.get_corner_points()

    pt_list = np.array([[0, 1], [0,2],[0, 4], [1,3], 
        [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]])
    pts_i = dset.model2img(pts, pose_gt)
    for i in range(12):
        p1 = pts_i[pt_list[i, 0], :]
        p2 = pts_i[pt_list[i, 1], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', lw=1)

    pts_i = dset.model2img(pts, pose_pred)
    for i in range(12):
        p1 = pts_i[pt_list[i, 0], :]
        p2 = pts_i[pt_list[i, 1], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=1)

    img = plt.imshow(im)
    if not os.path.exists(path): 
        os.makedirs(path)
    plt.savefig(os.path.join(path, str(index)+'.eps'))
    plt.close()  

def draw_bboxs(index, dset, im, pose_preds, path):
    # pose_gt = NED2EDNPose(pose_gt)
    # pose_pred = NED2EDNPose(pose_pred)
    pt_list = np.array([[0, 1], [0,2],[0, 4], [1,3], 
        [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]])
    for model_index, pose_pred in pose_preds.items():
        
        obj_id = dset.obj_ids[model_index]
        obj = dset.models[obj_id]
        pts = obj.get_corner_points()
        pts_i = dset.model2img(pts, pose_pred)
        for i in range(12):
            p1 = pts_i[pt_list[i, 0], :]
            p2 = pts_i[pt_list[i, 1], :]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=1)

    img = plt.imshow(im)
    if not os.path.exists(path): 
        os.makedirs(path)
    plt.savefig(os.path.join(path, str(index)+'.eps'))
    plt.close()  


def draw_mask(im, mask, coords):
    C, H, W = im.shape
    res = torch.zeros(H, W)
    # mask = torch.zeros_like(mask)
    f_index = mask==1
    f_coords = coords[f_index, :]

    if f_coords.size(0) == 0:
        return res[None, :]
    else:
        res[f_coords[:, 1], f_coords[:, 0]] = 1
        return res[None, :]

class Logger():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses 'tensorboardX' for display
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.name = opt.name
        self.saved = True

        self.base_dir, self.train_dir, self.val_dir = create_folder(opt)

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            from tensorboardX import SummaryWriter
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.base_dir, 'runs', timestr)
            if not os.path.exists(path):
                os.makedirs(path)
            self.writer = SummaryWriter(path)

        # create a logging file to store training losses
        self.log_name = os.path.join(self.base_dir, opt.phase+'_loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, len_loader):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        batches_done = epoch * len_loader + iters
        epochs_total = self.opt.epoch_end 
        batches_left = epochs_total * len_loader - batches_done

        time_left = datetime.timedelta(seconds=batches_left * t_comp)

        message = '[epoch: %d/%d] [iters: %d/%d] [time: %.3f] [ETA: %s] ' % \
            (epoch, epochs_total, iters, len_loader, t_comp, time_left)

        for k, v in losses.items():
            message += '[%s %.3f] ' % (k, v)
            if self.opt.display_id > 0:
                self.writer.add_scalar('train/'+k, v, batches_done)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def print_current_metrics(self, epoch, iters, len_loader, metrics, is_train=True):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        batches_done = epoch * len_loader + iters
 
        if is_train:
            epochs = (self.opt.epoch_end - self.opt.epoch_start + 1)
        else:
            epochs = 0
        message = '[epoch: %d/%d] [iters: %d/%d] ' % \
            (epoch, epochs, iters, len_loader)

        for key in metrics.metrics:
            message += metrics.message(key) 
            message += '\t\t\t\t'
            if self.opt.display_id > 0:
                if is_train:
                    self.writer.add_scalar('train/'+key+'/rmse', metrics.metrics[key].rmse, batches_done)
                else:
                    self.writer.add_scalar('test/'+key+'/rmse', metrics.metrics[key].rmse, batches_done)

        print(message, end="\r")  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_statistics(self, results, epoch=0, iters=0, len_loader=0, is_train=True):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if is_train:
            batches_done = epoch * len_loader + iters
            epochs = (self.opt.epoch_end - self.opt.epoch_start + 1)
            
            message = 'test: [epoch: %d/%d] [iters: %d/%d] ' % \
                (epoch, epochs, iters, len_loader)
        else:
            batches_done = 0
            message = 'test: '

        for key, val in results.items():
            message += ' [%s: %.3f]' % (key, val) 
            # message += '\t'
            if self.opt.display_id > 0:
                if is_train:
                    self.writer.add_scalar('train/'+key, val, batches_done)
                else:
                    self.writer.add_scalar('test/'+key, val, batches_done)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_success_rate(self, num_models, total_instance_cnt, 
        success_cnt, obj_ids, obj_dics):
       
        message = 'test: success rate: '
        for i in range(num_models):
            if total_instance_cnt[i] != 0:
                model_id = obj_ids[i]
                model_name = obj_dics[model_id]
                message += (' [%s: %.3f]' % (
                    model_name, float(success_cnt[i]) / total_instance_cnt[i]))
        message += '\n'
        message += ('\toveral success rate: %.3f' % (
            float(sum(success_cnt)) / sum(total_instance_cnt)))
        message += '\n'
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def save_and_display_images(self, stage, epoch, iters, len_loader, rgb_images, 
        gt_seg, pred_seg, coord):
        """save generated images; also display the images on tensorboard

        Parameters:
            stage (str) -- 'train' or 'test'
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            len_loader (int) -- total number of images in the dataset
            rgb_images (list) -- rgb images
            depth_images (list) -- depth_images
        """
        # bs = min(rgb_images[0].size(0), 2)
        bs = 1
        batches_done = epoch * len_loader + iters
        output_dir = self.train_dir if stage=='train' else self.val_dir

        # img_sample = torch.cat((
        #     scale_to_255(rgb_images[0][:bs,:,:,:].data.cpu()),
        #     colorize_depthmap_batch(depth_images[0][:bs,:,:]),
        #     colorize_depthmap_batch(depth_images[1][:bs,:,:]),
        # ), 0)
        img = torch.from_numpy(rgb_images[0]).permute(2, 0, 1).float()

        gt_seg_vis = draw_mask(img, gt_seg, coord)
        pred_seg_vis = draw_mask(img, pred_seg, coord)
        
        img_rgb = img[None, :]
        img_mask = torch.cat((
            colorize_depthmap_batch(gt_seg_vis),
            colorize_depthmap_batch(pred_seg_vis),
        ), 0)

        torchvision.utils.save_image(img_rgb,
            os.path.join(output_dir, '%02d-%05d.png' % (epoch, iters)),
            nrow=1, normalize=True)
        torchvision.utils.save_image(img_mask,
            os.path.join(output_dir, 'mask_%02d-%05d.png' % (epoch, iters)),
            nrow=1, normalize=True)
    

        if self.display_id > 0:    
            img_rgb = torchvision.utils.make_grid(img_rgb, nrow=bs, normalize=True)
            img_mask = torchvision.utils.make_grid(img_mask, nrow=bs, normalize=True)
            self.writer.add_image(stage+'/rgb', img_rgb, batches_done)
            self.writer.add_image(stage+'/mask', img_mask, batches_done)
            # self.writer.add_image(stage + '/rgb_masked', rgb_images[1], batches_done)
            # self.writer.add_image(stage + '/depth', depth_images[0], batches_done)
            # self.writer.add_image(stage + '/depth_pred', depth_images[1], batches_done)


    def vis_images(self, stage, epoch, iters, len_loader, rgb_images, 
        mask_images):
        """save generated images; also display the images on tensorboard

        Parameters:
            stage (str) -- 'train' or 'test'
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            len_loader (int) -- total number of images in the dataset
            rgb_images (list) -- rgb images
            depth_images (list) -- depth_images
        """
        bs = min(rgb_images[0].size(0), 2)
        batches_done = epoch * len_loader + iters
        output_dir = self.train_dir if stage=='train' else self.val_dir

        # img_sample = torch.cat((
        #     scale_to_255(rgb_images[0][:bs,:,:,:].data.cpu()),
        #     colorize_depthmap_batch(depth_images[0][:bs,:,:]),
        #     colorize_depthmap_batch(depth_images[1][:bs,:,:]),
        # ), 0)
        img_rgb = rgb_images[0]

        img_mask = torch.cat((
            colorize_depthmap_batch(mask_images[0]),
            colorize_depthmap_batch(mask_images[1]),
        ), 0)

        torchvision.utils.save_image(img_rgb,
            os.path.join(output_dir, '%02d-%05d.png' % (epoch, iters)),
            nrow=1, normalize=True)
        torchvision.utils.save_image(img_mask,
            os.path.join(output_dir, 'mask_%02d-%05d.png' % (epoch, iters)),
            nrow=1, normalize=True)
    
        if self.display_id > 0:    
            img_rgb = torchvision.utils.make_grid(img_rgb, nrow=bs, normalize=True)
            img_mask = torchvision.utils.make_grid(img_mask, nrow=bs, normalize=True)
            self.writer.add_image(stage+'/rgb', img_rgb, batches_done)
            self.writer.add_image(stage+'/mask', img_mask, batches_done)
