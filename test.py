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
    parser.add_argument('-c', '--config', type=str, default='config/config.json',
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
    # Convert to NoneDict, which return None for missing key.
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

    # dataset
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
    
    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  (Img_name, val_data) in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        mk_img = Metrics.tensormask2img(visuals['MASK'])

        sr_img_mode = 'grid'

        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])

        result_hr_path = os.path.join(result_path, 'HR')
        os.makedirs(result_hr_path, exist_ok=True)
        Metrics.save_img(
            hr_img, '{}/{}.jpg'.format(result_hr_path, Img_name[0]))

        result_sr_path = os.path.join(result_path, 'SR')
        os.makedirs(result_sr_path, exist_ok=True)
        Metrics.save_img(
            sr_img, '{}/{}.jpg'.format(result_sr_path, Img_name[0]))

        result_mask_path = os.path.join(result_path, 'MASK')
        os.makedirs(result_mask_path, exist_ok=True)
        Metrics.save_img(
            mk_img, '{}/{}.jpg'.format(result_mask_path, Img_name[0]))

        eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
        eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)

        avg_psnr += eval_psnr
        avg_ssim += eval_ssim
        print(f"ID: {idx}; NAME {Img_name[0]}; PSNR: {eval_psnr}; SSIM: {eval_ssim}")

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    logger_val = logging.getLogger('val')
    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        current_epoch, current_step, avg_psnr, avg_ssim))

    if wandb_logger:
        if opt['log_eval']:
            wandb_logger.log_eval_table()
        wandb_logger.log_metrics({
            'PSNR': float(avg_psnr),
            'SSIM': float(avg_ssim)
        })
