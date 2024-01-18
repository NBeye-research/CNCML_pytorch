import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
#early stopping
from pytorchtools import EarlyStopping


def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '-' + config['model']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join(config['output'], config['model_args']['encoder'], svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    print('begin loadd train sample. n_train_way:{}, n_train_shot:{}, n_query:{}, n_way:{}, n_shot:{}'.format(n_train_way, n_train_shot, n_query, n_way, n_shot))
    train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, config['val_batches'],
            n_way, n_shot + n_query,
            ep_per_batch=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    utils.log('train dataset: {} (x{}), {}, {}'.format(
            train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes, train_dataset.id2classes))
    utils.log('val dataset: {} (x{}), {}, {}'.format(
            val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes, val_dataset.id2classes))

    ########

    #### Model and optimizer ####
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])
        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())
    # print(model)

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    ########
    
    max_epoch = config['max_epoch']
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True, path='')

    trained_sample_num_2_loss = []
    sample_num = 0
    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_losses_every_batch = []
        train_acc_every_batch = []
        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()

            logits = model(x_shot, x_query).view(-1, n_train_way)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            train_losses_every_batch.append(loss.item())
            train_acc_every_batch.append(acc)
            sample_num += len(logits)
            trained_sample_num_2_loss.append(str(sample_num) + "\t" + str(loss.item()))

            logits = None; loss = None 

        train_loss = np.average(train_losses_every_batch)
        train_acc = np.average(train_acc_every_batch)
        # eval
        model.eval()
        val_losses_every_batch = []
        val_acc_every_batch = []

        np.random.seed(0)
        for data, _ in tqdm(val_loader, desc='val', leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=4)
            label = fs.make_nk_label(n_way, n_query,
                    ep_per_batch=4).cuda()

            with torch.no_grad():
                logits = model(x_shot, x_query).view(-1, n_way)
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)
            
            aves['vl'].add(loss.item())
            aves['va'].add(acc)
            val_losses_every_batch.append(loss.item())
            val_acc_every_batch.append(acc)

        valid_loss = np.average(val_losses_every_batch)
        valid_acc = np.average(val_acc_every_batch)
        print('average evaluation: epoch {}, train loss {:.4f}, train acc {:.4f}, '
                'val loss {:.4f},val acc {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc))

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.counter == 0:
            if epoch <= max_epoch:
                
                torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
                if aves['va'] > max_va:
                    max_va = aves['va']
                    torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
            else:
                torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))
        if early_stopping.early_stop:
            print('trigger early stop.')
            break

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_meta.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default='smartphone_data_aug')
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

