import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler

from sklearn import metrics
import codecs
from scipy import stats
from sklearn.utils import resample
import math

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

def compute_accuracy_confidence_interval(y_true, y_pred):

    n = len(y_true)
    k = np.sum(y_true == y_pred)

    accuracy = k / n
    
    current_acc_ci_low = accuracy - 1.96 * math.sqrt((accuracy*(1-accuracy))/n)
    # current_acc_ci_low = format(current_acc_ci_low, '.3f')
    current_acc_ci_up = accuracy + 1.96 * math.sqrt((accuracy*(1-accuracy))/n)

    # mu = n * accuracy
    # sigma = np.sqrt(n * accuracy * (1-accuracy))
    # confidence_interval = stats.norm.interval(0.95, loc=mu, scale=sigma/n)

    return accuracy, current_acc_ci_low, current_acc_ci_up

def compute_auc_confidence_interval(y_true, y_scores, n_bootstrap=1000, alpha=0.05):

    auc_values = np.zeros(n_bootstrap)
    
    # resample
    for i in range(n_bootstrap):
       
        resampled_indices = resample(range(len(y_true)), replace=True)
        y_true_resampled = y_true[resampled_indices]
        y_scores_resampled = y_scores[resampled_indices]
        
        auc_values[i] = metrics.roc_auc_score(y_true_resampled, y_scores_resampled)
    
    lower_bound = np.percentile(auc_values, 100 * (alpha / 2))
    upper_bound = np.percentile(auc_values, 100 * (1 - alpha / 2))
    
    return lower_bound, upper_bound

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main(config):
    print('test dataset :', config['test_dataset_args']['root_path'])

    # dataset
    test_data_loader_dict = {}
    dataset_arr = datasets.make(config['test_dataset'], **config['test_dataset_args'])
    if not isinstance(dataset_arr, list):
        test_dataset_single = datasets.make(config['test_dataset'],
                                  **config['test_dataset_args'])
        test_loader_single = DataLoader(test_dataset_single, config['batch_size'], shuffle=False,
                                num_workers=8, pin_memory=True)
        test_data_loader_dict[test_dataset_single] = test_loader_single
    else:
        for dataset in dataset_arr:
            utils.log('dataset: {} shape:{} (x{}), {}, {}'.format(
                    dataset.rootpath, dataset[0][0].shape, len(dataset), dataset.n_classes, dataset.classes))
            test_loader_single = DataLoader(dataset, config['batch_size'], shuffle=False,
                                    num_workers=8, pin_memory=True)
            test_data_loader_dict[dataset] = test_loader_single
    print('total test set len:', len(test_data_loader_dict))

    if config.get('load'):
        print('load model from:', config['load'])
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)

    print('parameter number:', get_parameter_number(model))

    if config.get('_parallel'):
        model = nn.DataParallel(model)
    # print(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    
    ########   
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    acc_str = ''
    auc_str = ''
    model.eval()
    with torch.no_grad():
        for test_dataset in test_data_loader_dict:
            print('begin evaluation test set:', test_dataset.rootpath)
            test_loader = test_data_loader_dict[test_dataset]
            result = {}
            # test
            test_losses = []
            preds, labels, scores,features = [], [], [], []
            for data, label in tqdm(test_loader, desc='test', leave=False):
                data, label = data.cuda(), label.cuda()
                logits, encoder = model(data)
                features.append(encoder)
                # print(encoder.shape)
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)
                score = torch.softmax(logits, dim=1)
                predict = torch.max(logits, dim=1)[1]
                labels.append(label)
                scores.append(score)
                preds.append(predict)
            features = torch.cat(features, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            predicts = torch.cat(preds, dim=0).cpu().numpy()
            scores = torch.cat(scores, dim=0).cpu().numpy()
            report = metrics.classification_report(labels, predicts, target_names=['{}'.format(x) for x in range(test_dataset.n_classes)],
                                                    digits=4, labels=range(test_dataset.n_classes))
            
            confusion = metrics.confusion_matrix(labels, predicts)
            print(report)
            print(confusion)
            # performance = np.sum(labels==predicts) / len(labels)
            performance, acc_low_bound, acc_high_bound = compute_accuracy_confidence_interval(labels, predicts)

            print("accuracy:{:.2f}, ({:.2f}, {:.2f})".format(performance*100, acc_low_bound*100, acc_high_bound*100))
            
            # transfer true label to one-hot style.
            true_labels_onehot = label_binarize(labels, classes=np.arange(dataset.n_classes))
           
            auc_score_mean = roc_auc_score(true_labels_onehot, scores, multi_class='ovr')
            auc_score = roc_auc_score(true_labels_onehot, scores, multi_class='ovr', average=None)
            auc_low_bound, auc_high_bound = compute_auc_confidence_interval(true_labels_onehot, scores)
            print("AUC Score:{} AUC mean:{:.2f}, ({:.2f}, {:.2f})".format(auc_score, auc_score_mean*100, auc_low_bound*100, auc_high_bound*100))
            acc_str += '\t' + "{:.2f} ({:.2f}, {:.2f})".format(performance*100, acc_low_bound*100, acc_high_bound*100)
            auc_str += '\t' + "{:.2f} ({:.2f}, {:.2f})".format(auc_score_mean*100, auc_low_bound*100, auc_high_bound*100)
            # sensitivity and specificity
            for label_index in range(dataset.n_classes):
                sensitivity = calculate_sensitivity(confusion, label_index)
                specificity = calculate_specificity(confusion, label_index)

                print("\tClass", label_index)
                print("\tSensitivity:{:.2f}".format(sensitivity*100))
                print("\tSpecificity:{:.2f}".format(specificity*100))
                print()
                
    print(acc_str)
    print(auc_str)
    print('all test data predict finish.')
              


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_classifier.yaml')
    parser.add_argument('--gpu', default='2,3')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

