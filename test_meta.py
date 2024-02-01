import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from sklearn import metrics
import codecs


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def calculate_sensitivity(confusion_matrix, class_index):
    TP = confusion_matrix[class_index, class_index]
    FN = np.sum(confusion_matrix[class_index, :]) - TP
    sensitivity = TP / (TP + FN)
    return sensitivity

def calculate_specificity(confusion_matrix, class_index):
    TN = np.sum(confusion_matrix) - np.sum(confusion_matrix[class_index, :]) - np.sum(confusion_matrix[:, class_index]) + confusion_matrix[class_index, class_index]
    FP = np.sum(confusion_matrix[:, class_index]) - confusion_matrix[class_index, class_index]
    specificity = TN / (TN + FP)
    return specificity

def main(config):
    print('test dataset :', config['dataset_args']['root_path'])
    print('model:',config['load'])
    print('is nearest point:',config['nearest_point'])
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}, {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes, dataset.classes))
    if not args.sauc:
        n_way = config['n_way']
    else:
        n_way = 2
    # n_way = config['n_way']
    n_shot, n_query, n_shots = config['n_shot'], config['n_query'], config['n_shots']
    n_batch = config['test_batches']
    ep_per_batch = config['ep_per_batch']

    fs_loaders = []
    for tmp_shot in n_shots:
        fs_sampler = CategoriesSampler(
                dataset.label, n_batch, n_way, tmp_shot + n_query,
                ep_per_batch=ep_per_batch, isTest=True)
        fs_loader = DataLoader(dataset, batch_sampler=fs_sampler,
                                num_workers=8, pin_memory=True)
        fs_loaders.append(fs_loader)

    print('begin to generate sample. n_batch:{}, n_way:{}, n_shots:{}, n_query:{}, n_sample:{}, ep_per_batch:{}'.format(n_batch, n_way, n_shots, n_query, (n_shot + n_query),
            ep_per_batch))

    # model
    if config.get('load') is not None:
        print('load model from:', config['load'])
        model = models.load(torch.load(config['load']))
    else:
        print('model dir is None.')
        return

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    # print('model:', model)
    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    test_epochs = args.test_epochs
    np.random.seed(0)
   
    result = ['']*test_epochs
    for i, n_shot in enumerate(n_shots):
        # Set the timer
        timer_epoch = utils.Timer()
        va_lst = []
        aves_keys = ['vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}
        for epoch in range(1, test_epochs + 1):
            preds, labels, scores , aucs = [], [], [], [], []
            timer_epoch.s()
            print('begin to predict. i:{}, n_way:{}, n_shot:{}, n_query:{}'.format(i, n_way, n_shot, n_query))
            for data, _ in tqdm(fs_loaders[i], leave=False):
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query,
                        ep_per_batch=ep_per_batch)
                with torch.no_grad():
                    if not args.sauc:
                        logits = model(x_shot, x_query, config['nearest_point'])
                        logits = logits.view(-1, n_way)
                        label = fs.make_nk_label(n_way, n_query,
                                ep_per_batch=ep_per_batch).cuda()
                        loss = F.cross_entropy(logits, label)
                        acc = utils.compute_acc(logits, label)

                        aves['vl'].add(loss.item(), len(label))
                        aves['va'].add(acc, len(label))

                        va_lst.append(acc)
                        score = torch.softmax(logits, dim=1)
                        predict = torch.max(logits, dim=1)[1]
                        
                        tmp_labels_onehot = label_binarize(label.cpu().numpy(), classes=np.arange(dataset.n_classes))
                        tmp_score_mean = roc_auc_score(tmp_labels_onehot, score.cpu().numpy(), multi_class='ovr')
                        labels.append(label)
                        scores.append(score)
                        preds.append(predict)
                        aucs.append(tmp_score_mean)
                    else:
                        x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                        shot_shape = x_shot.shape[:-3]
                        img_shape = x_shot.shape[-3:]
                        bs = shot_shape[0]
                        p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                                *shot_shape, -1).mean(dim=1, keepdim=True)
                        q = model.encoder(x_query.view(-1, *img_shape)).view(
                                bs, -1, p.shape[-1])
                        p = F.normalize(p, dim=-1)
                        q = F.normalize(q, dim=-1)
                        s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                        for i in range(bs):
                            k = s.shape[1] // 2
                            y_true = [1] * k + [0] * k
                            acc = roc_auc_score(y_true, s[i])
                            aves['va'].add(acc, len(data))
                            va_lst.append(acc)
            t_epoch = utils.time_str(timer_epoch.t())
            labels = torch.cat(labels, dim=0).cpu().numpy()
            predicts = torch.cat(preds, dim=0).cpu().numpy()
            scores = torch.cat(scores, dim=0).cpu().numpy()
            # features = torch.cat(features, dim=0).cpu().numpy()
            performance = np.sum(labels==predicts) / len(labels)
            print('len labels', aves['va'].count())
            print('len labels', len(labels))
            report = metrics.classification_report(labels, predicts, target_names=['{}'.format(x) for x in range(dataset.n_classes)],
                                                    digits=4, labels=range(dataset.n_classes))
            confusion = metrics.confusion_matrix(labels, predicts)
            print('fs :{} test epoch: {}, report:\n{}'.format(n_shot, epoch, report))
            print('fs :{} test epoch: {}, confusion:\n{}'.format(n_shot, epoch, confusion))
            result[epoch - 1] += ' fs-{}:{:.2f} +- {:.2f}'.format( n_shot, aves['va'].item() * 100,
                    mean_confidence_interval(va_lst) * 100)
            
            true_labels_onehot = label_binarize(labels, classes=np.arange(dataset.n_classes))
            
            auc_score_mean = roc_auc_score(true_labels_onehot, scores, multi_class='ovr')
            auc_score = roc_auc_score(true_labels_onehot, scores, multi_class='ovr', average=None)
            print("fs :{} test epoch: {} AUC Score:{} AUC mean:{:.2f} +- {:.2f} ".format(n_shot, epoch, auc_score, auc_score_mean, mean_confidence_interval(aucs) * 100))

            for label_index in range(dataset.n_classes):
                sensitivity = calculate_sensitivity(confusion, label_index)
                specificity = calculate_specificity(confusion, label_index)
                print("fs :{} test epoch: ".format(n_shot, epoch))
                print("\tClass", label_index)
                print("\tSensitivity:", sensitivity)
                print("\tSpecificity:", specificity)
                print()
            
            print('fs-{} test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})， time consume={}'.format(
                    n_shot, epoch, aves['va'].item() * 100,
                    mean_confidence_interval(va_lst) * 100,
                    aves['vl'].item(), _[-1],t_epoch))
            print('fs-{} test epoch {}: acc= {:.2f} ({:.2f}, {:.2f}) auc= {:.2f} ({:.2f}, {:.2f})'.format(
                    n_shot, epoch, aves['va'].item() * 100,
                    aves['va'].item() * 100 - mean_confidence_interval(va_lst) * 100, aves['va'].item() * 100 + mean_confidence_interval(va_lst) * 100,
                    auc_score_mean * 100 , auc_score_mean* 100 - mean_confidence_interval(aucs) * 100, auc_score_mean* 100 + mean_confidence_interval(aucs) * 100))       

    for index in range(len(result)):
        print('epoch:{}'.format(index + 1))
        print(result[index].strip())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_meta.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=2)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0,1,2,3')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)

