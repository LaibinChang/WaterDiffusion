import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import wandb
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadow.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    best_ssim = 0.0
    best_psnr = 0.0  
    while current_step < n_iter:
        current_epoch += 1
        for _, (i, train_data) in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()

            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics(logs)

            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0
                result_path = '{}/{}'.format(opt['path']
                                                ['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')
                for _,  (Img_name, val_data) in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    sr_img = Metrics.tensor2img(visuals['SR']) 
                    mk_img = Metrics.tensormask2img(visuals['MASK']) 
                    hr_img = Metrics.tensor2img(visuals['HR']) 

                    Metrics.save_img(
                        mk_img, '{}/{}_mask.jpg'.format(result_path, Img_name[0]))
                    Metrics.save_img(
                        sr_img, '{}/{}_sr.jpg'.format(result_path, Img_name[0]))

                    avg_psnr += Metrics.calculate_psnr(
                        sr_img, hr_img)
                    avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)

                    if wandb_logger:
                        wandb_logger.log_image(
                            f'validation_{idx}',
                            np.concatenate((sr_img, hr_img), axis=1)
                        )

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')
                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                    current_epoch, current_step, avg_psnr, avg_ssim))
                # tensorboard logger
                tb_logger.add_scalar('psnr', avg_psnr, current_step)
                tb_logger.add_scalar('ssim', avg_ssim, current_step)
                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/val_psnr': avg_psnr,
                        'validation/val_ssim': avg_ssim,
                        'validation/val_step': val_step
                    })
                    val_step += 1
                
                if avg_ssim > best_ssim:  
                    best_ssim = avg_ssim  
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
                    
                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)
                        
                if avg_psnr > best_psnr:  
                    best_psnr = avg_psnr  
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
                    
                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)
                    
        if wandb_logger:
            wandb_logger.log_metrics({'epoch': current_epoch-1})

    logger.info('End of training.')
