"""
Parts of code taken from https://github.com/microsoft/MeshTransformer
"""

import torch
import time
import datetime
import os.path as path
import os
import cv2
import numpy as np
import json

from utils.metric_logger import AverageMeter
from utils.utils import createDir
from . import mano_config as cfg
from utils.geometric_layers import orthographicProjection
from utils.render import visualizeMesh

def saveCheckpoint(model, args, epoch, iteration, optimizer, scaler, num_trial=10):
    if args.checkpoint_dir is not None:
        checkpoint_dir = path.join(args.checkpoint_dir, 'checkpoint-{}-{}'.format(
            epoch, iteration))
    else:
        checkpoint_dir = path.join(args.output_dir, 'checkpoint-{}-{}'.format(
            epoch, iteration))

    createDir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, path.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), path.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, path.join(checkpoint_dir, 'training_args.bin'))
            torch.save(optimizer.state_dict(), path.join(checkpoint_dir, 'opt_state_dict.bin'))
            torch.save(scaler.state_dict(), path.join(checkpoint_dir, 'scaler_state_dict.bin'))
            args.logger.createLog("SAVE_CHECKPOINT", "Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass

    if i>=num_trial:
        args.logger.createLog("SAVE_CHECKPOINT", "Failed to save checkpoint after {} trails.".format(num_trial))
    
    return checkpoint_dir

# After half epochs decrease learning rate
def adjustLearningRate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.learning_rate * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
#---------------------------------LOSSES FUNCTIONS---------------------------------
def joints2dLoss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def joints3dLoss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d=True):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

def verticesLoss(criterion_vertices, pred_vertices, gt_vertices, has_smpl=True):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()


def poseLoss(criterion_pose, pred_pose, gt_pose):
    """
    Compute pose parameters loss 
    """
    return criterion_pose(pred_pose, gt_pose.unsqueeze(-1))
    

def betasLoss(criterion_betas, pred_betas, gt_betas):
    """
    Compute betas parameters loss 
    """

    return criterion_betas(pred_betas, gt_betas.unsqueeze(-1))


#---------------------------------RUN FUNCTIONS---------------------------------
def run(args, train_dataloader, TransRecon_model, mano_model, renderer, mesh_sampler):

    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs

    optimizer = torch.optim.Adam(params=list(TransRecon_model.parameters()),
                                           lr=args.learning_rate,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
    scaler = torch.cuda.amp.GradScaler()
    
    # Define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').to(args.device)
    criterion_joints = torch.nn.MSELoss(reduction='none').to(args.device)
    #criterion_vertices = torch.nn.L1Loss().to(args.device)
    #criterion_pose = torch.nn.L1Loss().to(args.device)
    #criterion_betas = torch.nn.L1Loss().to(args.device)

    start_training_time = time.time()
    end = time.time()
    
    TransRecon_model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    #log_loss_pose = AverageMeter()
    #log_loss_betas = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    #log_loss_vertices = AverageMeter()
    
    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):
        
        iteration += 1 + args.iteration_restart
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjustLearningRate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.cuda()

        gt_pose = annotations['pose'].cuda()
        gt_betas = annotations['betas'].cuda()
        gt_2d_joints = annotations['joints_2d'].cuda()

        has_mesh = annotations['has_smpl'].cuda()
        has_3d_joints = has_mesh
        #has_2d_joints = has_mesh

        #mpm_mask = annotations['mpm_mask'].cuda()
        mjm_mask = annotations['mjm_mask'].cuda()

        # Generate mesh from pose and betas
        gt_vertices, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
        gt_vertices = gt_vertices/1000.0
        gt_3d_joints = gt_3d_joints/1000.0

        #gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

        # Normalize ground truth based on hand's root 
        gt_3d_root = gt_3d_joints[:, cfg.ROOT_INDEX, :]
        gt_vertices = gt_vertices - gt_3d_root[:, None, :]
        #gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
        gt_3d_joints_with_tag = torch.ones((batch_size, gt_3d_joints.shape[1], 4)).cuda()
        gt_3d_joints_with_tag[:, :, :3] = gt_3d_joints

        # Prepare masks for 3d joints modeling
        #mpm_mask_ = mpm_mask.expand(-1, -1, 576)
        mjm_mask_ = mjm_mask.expand(-1, -1, 576)
        #meta_masks = torch.cat([mpm_mask_, mbm_mask_], dim=1)
        
        # Forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, pred_pose, pred_betas = TransRecon_model(images, mano_model, mesh_sampler, 
                                                                                                               meta_masks=meta_masks, is_train=True)

        # Regress 3d joints from the mesh
        pred_3d_joints_from_mesh = mano_model.get3dJointsFromMesh(pred_vertices)

        # Obtain 2d joints from regressed ones and from MANO ones
        pred_2d_joints_from_mesh = orthographicProjection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
        pred_2d_joints = orthographicProjection(pred_3d_joints.contiguous(), pred_camera.contiguous())

        # Compute 3d joint loss 
        loss_3d_joints = joints3dLoss(criterion_joints, pred_3d_joints, gt_3d_joints_with_tag, has_3d_joints)

        # Compute 3d vertices loss
        loss_vertices = verticesLoss(criterion_vertices, pred_vertices, gt_vertices, has_mesh)

        # Compute pose and betas losses
        #loss_pose = poseLoss(criterion_pose, pred_pose, gt_pose)
        #loss_betas = betasLoss(criterion_betas, pred_betas, gt_betas)

        # Compute 3d regressed joints loss 
        loss_reg_3d_joints = joints3dLoss(criterion_joints, pred_3d_joints_from_mesh, gt_3d_joints_with_tag, has_3d_joints)
        # Compute 2d joints loss
        loss_2d_joints = joints2dLoss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints)  + \
                         joints2dLoss(criterion_2d_keypoints, pred_2d_joints_from_mesh, gt_2d_joints)

        loss_3d_joints = loss_3d_joints + loss_reg_3d_joints

        loss = args.joints_loss_weight * loss_3d_joints + \
               args.vertices_loss_weight * loss_vertices + \
               args.vertices_loss_weight * loss_2d_joints 

        # Update logs
        #log_loss_pose.update(loss_pose.item(), batch_size)
        #log_loss_betas.update(loss_betas.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_losses.update(loss.item(), batch_size)
        
        # Backward-pass
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)

        # Updates the scale for next iteration.
        #scaler.update()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            args.logger.createLog("RUN",
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, compute time avg: {:.4f}, data time avg: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_3djoints.avg, log_loss_vertices.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )
                
        # Save a checkpoint and visualize partial results obtained
        if iteration % iters_per_epoch == 0:
            if epoch%5 == 0:
                saveCheckpoint(TransRecon_model, args, epoch, iteration, optimizer, scaler)
                visual_imgs = visualizeMesh(renderer,
                                            annotations['ori_img'].detach(),
                                            annotations['joints_2d'].detach(),
                                            pred_vertices.detach(), 
                                            pred_camera.detach(),
                                            pred_2d_joints_from_mesh.detach())
                #visual_imgs = visual_imgs.transpose(0,1)
                #visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = torch.einsum("abc -> bca", visual_imgs)
                visual_imgs = np.asarray(visual_imgs)

                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:, :, ::-1]*255))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    args.logger.createLog("RUN", 'Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    saveCheckpoint(TransRecon_model, args, epoch, iteration)

def runInferenceHandMesh(args, val_loader, TransRecon_model, mano_model, mesh_sampler, renderer):
    TransRecon_model.eval()
    
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []
    
    with torch.no_grad():
        for idx, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            
            images = images.cuda()

            # Make a forward-pass to inference
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, pred_pose, pred_betas = TransRecon_model(images, mano_model, mesh_sampler)

            # Take 3d joints from full mesh
            pred_3d_wrist = pred_3d_joints[:, cfg.ROOT_INDEX, :]
            pred_3d_joints_from_mesh = pred_3d_joints - pred_3d_wrist[:, None, :]
            pred_vertices = pred_vertices - pred_3d_wrist[:, None, :]

            for batch in range(batch_size):
                fname_output_save.append(img_keys[batch])
                
                mesh_output_save.append(pred_vertices[batch].tolist())

                joint_output_save.append(pred_3d_joints[batch].tolist())

            if idx%20==0:
                # Obtain 3d joints, which are regressed from the full mesh
                pred_3d_joints_from_mesh = mano_model.get3dJointsFromMesh(pred_vertices)
                # Get 2d joints from orthographic projection of 3d ones taken from mesh
                pred_2d_joints_from_mesh = orthographicProjection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
                
                # Transform mesh into image
                visual_imgs = visualizeMesh(renderer,
                                             annotations['ori_img'].detach(),
                                             annotations['joints_2d'].detach(),
                                             pred_vertices.detach(), 
                                             pred_camera.detach(),
                                             pred_2d_joints_from_mesh.detach())

                #visual_imgs = visual_imgs.transpose(0, 1)
                #visual_imgs = visual_imgs.transpose(1, 2)
                visual_imgs = torch.einsum(visual_imgs, "abc -> bca")
                visual_imgs = np.asarray(visual_imgs)
                
                inference_setting = "scale{02d}_rot{s}".format(int(args.sc*10), str(int(args.rot)))
                temp_fname = args.output_dir + args.saved_checkpoint[0:-9] + "freihand_results_"+inference_setting+"_batch"+str(idx)+".jpg"
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:, :, ::-1]*255))

    # Saving predictions into a zip file
    args.logger.createLog("RUN_INFERENCE", "---------Saving results to 'pred.json'---------")
    
    with open("pred.json", "w") as f:
        json.dump([joint_output_save, mesh_output_save], f)

    run_exp_name = args.saved_checkpoint.split('/')[-3]
    run_ckpt_name = args.saved_checkpoint.split('/')[-2].split('-')[1]
    inference_setting = "scale{02d}_rot{s}".format(int(args.sc*10), str(int(args.rot)))
    resolved_submit_cmd = "zip " + args.output_dir + run_exp_name + "-ckpt"+ run_ckpt_name + "-" + inference_setting +"-pred.zip " +  "pred.json"
    
    args.logger.createLog("RUN_INFERENCE", "---------Executing: {}---------".format(resolved_submit_cmd))
    os.system(resolved_submit_cmd)
    
    resolved_submit_cmd = 'rm pred.json'
    args.logger.createLog("RUN_INFERENCE", "Executing: {}".format(resolved_submit_cmd))
    os.system(resolved_submit_cmd)
    
    return

def runEvalAndSave(args, split, val_dataloader, TransRecon_model, mano_model, renderer, mesh_sampler):

    #criterion_keypoints = torch.nn.MSELoss(reduction='none').to(args.device)
    #criterion_vertices = torch.nn.L1Loss().to(args.device)

    runInferenceHandMesh(args, val_dataloader, TransRecon_model, 
                         mano_model, mesh_sampler, renderer)
    
    saveCheckpoint(TransRecon_model, args, 0, 0)
    
    return
