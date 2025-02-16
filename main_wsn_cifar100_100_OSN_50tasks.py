# Authorized by Haeyong Kang.

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random

import argparse,time
import math
from copy import deepcopy
from itertools import combinations, permutations
from torch.autograd import Variable
from utils import *
from flatness_minima import SAM
from networks.subnet import SubnetLinear, SubnetConv2d
from networks.alexnet import SubnetAlexNet_norm as AlexNet
from networks.lenet import SubnetLeNet as LeNet
from networks.utils import *

from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

import importlib
from utils_rle import compress_ndarray, decompress_ndarray, comp_decomp_mask
from utils_huffman import comp_decomp_mask_huffman

def train(args, model, device, x, y, optimizer, criterion, task_id_nominal, consolidated_masks):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if ((i + args.batch_size_train) <= len(r)):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]

        data = x[b]
        data, target = data.to(device), y[b].to(device)

        N = data.shape[0]
        lam = (beta_distributions(size=N, alpha=args.mixup_alpha)).astype(np.float32)
        lam_adv = Variable(torch.from_numpy(lam)).to(device)
        lam_adv = torch.clamp(lam_adv, 0, 1)  # clamp to range [0,1)
        lam_adv.requires_grad = True
        index = torch.randperm(N).cuda()
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(data, lam_adv, target, index)
        
        output1 = model(data, task_id_nominal, mask=None, mode="train")
        output2 = model(mix_inputs, task_id_nominal, mask=None, mode='train')
        # output1 = model(data)[task_id_nominal]
        # output2 = model(mix_inputs)[task_id_nominal]
        loss = criterion(output1, target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        grad_lam_adv = lam_adv.grad.data
        grad_norm = torch.norm(grad_lam_adv, p=2) + 1.e-16
        lam_adv.data.add_(grad_lam_adv * 0.05 / grad_norm)  # gradient assend by SAM
        lam_adv = torch.clamp(lam_adv, 0, 1)
        optimizer.perturb_step()

        # Weight Descent Step
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(data, lam_adv, target, index)
        mix_inputs = mix_inputs.detach()
        lam_adv = lam_adv.detach()

        output1 = model(data, task_id_nominal, mask=None, mode="train")
        output2 = model(mix_inputs, task_id_nominal, mask=None, mode='train')

        # output1 = model(data)[task_id_nominal]
        # output2 = model(mix_inputs)[task_id_nominal]
        loss = criterion(output1, target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()     

        # optimizer.zero_grad()
        # output = model(data, task_id_nominal, mask=None, mode="train")
        # loss = criterion(output, target)
        # loss.backward()

        # Continual Subnet no backprop
        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():

                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else: # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0.05 * getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] 

        optimizer.step()

def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id,consolidated_masks):
    fea = {}
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if ((i+args.batch_size_train) <= len(r)): 
            b = r[i : i + args.batch_size_train]
        else: 
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)

        N = data.shape[0]
        lam = (beta_distributions(size=N, alpha=args.mixup_alpha)).astype(np.float32)
        lam_adv = Variable(torch.from_numpy(lam)).to(device)
        lam_adv = torch.clamp(lam_adv, 0, 1)  # clamp to range [0,1)
        lam_adv.requires_grad = True       

        index = torch.randperm(N).cuda()
        # initialize x_mix
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(data, lam_adv, target, index)

        # Weight and Data Ascent Step
        output1 = model(data, task_id, mask=None, mode="train")
        output2 = model(mix_inputs, task_id, mask=None, mode='train')
        # output1 = model(data)[task_id]
        # output2 = model(mix_inputs)[task_id]
        loss = criterion(output1, target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        grad_lam_adv = lam_adv.grad.data
        grad_norm = torch.norm(grad_lam_adv, p=2) + 1.e-16
        lam_adv.data.add_(grad_lam_adv * 0.05 / grad_norm)  # gradient assend by SAM
        lam_adv = torch.clamp(lam_adv, 0, 1)
        optimizer.perturb_step()

        # Weight Descent Step
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(data, lam_adv, target, index)
        mix_inputs = mix_inputs.detach()
        lam_adv = lam_adv.detach()
        output1 = model(data, task_id, mask=None, mode="train")
        output2 = model(mix_inputs, task_id, mask=None, mode='train')
        # output1 = model(data)[task_id]
        # output2 = model(mix_inputs)[task_id]
        loss = criterion(output1, target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()


        # optimizer.zero_grad()        
        # output = model(data,task_id,mask=None,mode="train")
        # loss = criterion(output, target)         
        # loss.backward()

        # no gradient subnet
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():

                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else: # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        fea[key] = (getattr(getattr(model, module_name), module_attr))        

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 0] = getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 0] * 0       
        # Gradient Projections 
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<10 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                        feature_mat[int(kk / 2)]).view(params.size())
                kk +=1
            elif (k<10 and len(params.size())==1) and task_id !=0 :
                params.grad.data.fill_(0)

        optimizer.step()


def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test", comp_flag=False):
    model.eval()
    comp_ratio = 0
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    r=torch.LongTensor(r).to(device)

    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if ((i + args.batch_size_test) <= len(r)):
                b=r[i:i+args.batch_size_test]
            else: b=r[i:]

            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc, comp_ratio

def main(args):
    tstart=time.time()
    ## Device Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Load CIFAR100_100 DATASET
    dataloader = importlib.import_module('dataloader.' + args.dataset)

    data, output_info, input_size, n_tasks, n_outputs = dataloader.get_50tasks(data_path=args.data_path, args=args, seed=args.seed, pc_valid=args.pc_valid, samples_per_task=args.samples_per_task)
    args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))

    # Shuffle tasks
    if args.shuffle_task:
        ids = list(shuffle(np.arange(args.n_tasks), random_state=args.seed))
    else:
        ids = list(np.arange(args.n_tasks))

    print('Task info =', output_info)
    print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
    print('Task order =', ids)
    print('-' * 100)
    
    dist_tensor = torch.zeros(1,9)
    new_dist_list = []
    out_pre_list = []
    out_cur_list = []
    dist_list = []
    tasks_ = [t for t  in ids]
    n_outputs_ = [n_outputs] * n_tasks
    taskcla = [(t,n) for t,n in zip(tasks_, n_outputs_)]
    feature_pre = []
    acc_matrix=np.zeros((50,50))
    sparsity_matrix = []
    sparsity_per_task = {}
    task_id_sparsity = 0
    criterion = torch.nn.CrossEntropyLoss()
    # Model Instantiation
    if args.model == "alexnet":
        model = AlexNet(taskcla, args.sparsity,task_id_sparsity).to(device)
    elif args.model == "lenet":
        model = LeNet(taskcla, args.sparsity).to(device)
    else:
        raise Exception("[ERROR] The model " + str(args.model) + " is not supported!")
    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    per_task_masks, consolidated_masks, prime_masks = {},{},{}
    for k, ncla in taskcla:
        threshold = np.array([args.gpm_thro] * 5)
        #threshold = np.array([0.97] * 5) + task_id*np.array([0.003] * 5)
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        best_loss=np.inf
        print('-'*40)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print('-'*40)

        best_model=get_model(model)
        # if args.optim == "base":
        #     optimizer = optim.SGD(model.parameters(), lr=lr)
        # elif args.optim == "adam":
        #     optimizer = optim.Adam(model.parameters(), lr=lr)
        # else:
        #     raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")
        # initial_optimizer_state_dict = optimizer.state_dict()

        # reinitialized weight score
        # model.init_masks(task_id=k)
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = SAM(base_optimizer, model)
        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()
            train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks)
            clock1 = time.time()
            tr_loss,tr_acc, comp_ratio = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc, comp_ratio = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=get_model(model)
                patience=args.lr_patience
                print('*',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print('')
                        lr = args.lr_min
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer.optimizer, epoch, args)
            print('')
        
        # Restore best model
        set_model_(model,best_model)
        per_task_masks[task_id] = model.get_masks(task_id)

        # Consolidate task masks to keep track of parameters to-update or not
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        if task_id == 0:
            consolidated_masks = deepcopy(per_task_masks[task_id])
        else:
            for key in per_task_masks[task_id].keys():
                # Skip output head from other tasks
                # Also don't consolidate output head mask after training on new tasks; continue
                if "last" in key:
                    if key in curr_head_keys:
                        consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
                    continue
                # Or operation on sparsity
                if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                    consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))

        if task_id == 0:
            feature_list =[]
            # if consolidated_masks is not None and consolidated_masks != {}: 
            #     for key in consolidated_masks.keys():
            #         if 'last' in key:
            #             if key not in curr_head_keys:
            #                 continue
            #         if (len(key.split('.')) == 3): 
            #             module_name, task_num, module_attr = key.split('.')
            #         else: 
            #             module_name, module_attr = key.split('.')
            #         if (hasattr(getattr(model, module_name), module_attr)):
            #             if (getattr(getattr(model, module_name), module_attr) is not None):
            #                 with torch.no_grad():
            #                     getattr(getattr(model, module_name), module_attr)[consolidated_masks[key] == 1.0] *= 0
            mat_list = get_representation_matrix(model,device,xtrain,ytrain)
            feature_list = update_GPM(model,mat_list,threshold,feature_list)
            set_model_(model,best_model)
        else:
            best_model=get_model(model)
            base_optimizer = optim.Adam(model.parameters(), lr=args.lr)
            optimizer = SAM(base_optimizer, model)
            #set_model_(model,o_model)
            feature_mat = []
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                #Uf = Uf  + 0.05 * torch.eye(Uf.shape[0]).to(device)
                print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            print ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                clock0=time.time()
                train_projected(args,model,device,xtrain,ytrain,optimizer,criterion,feature_mat,task_id,consolidated_masks)
                clock1=time.time()
                tr_loss, tr_acc, comp_ratio= test(args, model, device, xtrain, ytrain,  criterion,task_id, curr_task_masks=consolidated_masks, mode="valid")
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc,comp_ratio = test(args, model, device, xvalid, yvalid,  criterion,task_id, curr_task_masks=None, mode="valid")
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                print('')

            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=get_model(model)
                patience=args.lr_patience
                print('*',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        lr = args.lr_min                  
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer.optimizer, epoch, args)
            print('')
            # Test 
            set_model_(model,best_model)
            test_loss, test_acc,comp_ratio = test(args, model, device, xtest, ytest,  criterion,task_id, curr_task_masks=None, mode="valid")
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
                            # Memory Update 
                # if consolidated_masks is not None and consolidated_masks != {}: 
                #     for key in consolidated_masks.keys():
                #         if 'last' in key:
                #             if key not in curr_head_keys:
                #                 continue
                #         if (len(key.split('.')) == 3): 
                #             module_name, task_num, module_attr = key.split('.')
                #         else: 
                #             module_name, module_attr = key.split('.')
                #         if (hasattr(getattr(model, module_name), module_attr)):
                #             if (getattr(getattr(model, module_name), module_attr) is not None):
                #                 with torch.no_grad():
                #                     getattr(getattr(model, module_name), module_attr)[consolidated_masks[key] == 0.0] *= 0
                #                     #getattr(getattr(model, module_name), module_attr)[consolidated_masks[key] == 1.0] += 0.05
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)
            set_model_(model,best_model)
        # print Sparsity
        sparsity_per_layer = print_sparsity(consolidated_masks)
        all_sparsity = global_sparsity(consolidated_masks)
        print("Global Sparsity: {}".format(all_sparsity))
        sparsity_matrix.append(all_sparsity)
        sparsity_per_task[task_id] = sparsity_per_layer
    

        # Test
        print ('-'*40)
        test_loss, test_acc, comp_ratio = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            if jj < task_id:
                acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
            else:
                xtest =data[ii]['test']['x']
                ytest =data[ii]['test']['y']

                _, acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            jj +=1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(i_a + 1):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print('')

        # update task id
        task_id +=1

    save_name = "csnb_cifar100_10_{}_{}_SEED_{}_LR_{}_SPARSITY_{}".format(args.model,args.encoding, args.seed, args.lr, 1 - args.sparsity)
    if not args.prune_thresh == 0.25:
        save_name += "_prune_thresh_{}".format(args.prune_thresh)
    safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".acc", acc_matrix)
    safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".cap", sparsity_matrix)
    safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".spar", sparsity_per_task)
    safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".pertask", per_task_masks)
    safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".fullmask", consolidated_masks)
    torch.save(model.to("cpu"), "./results2/csnb_cifar100_10/" + save_name + ".ptmodel")

    model = model.to(device)
    # # Test one more time
    # test_acc_matrix=np.zeros((10,10))
    # sparsity_matrix = []
    # mask_comp_ratio = []
    # sparsity_per_task = {}
    # criterion = torch.nn.CrossEntropyLoss()

    # task_list = []
    # task_id=0
    # for k, ncla in taskcla:
    #     print('*'*100)
    #     print('Task {:2d} ({:s})'.format(k,data[k]['name']))
    #     print('*'*100)
    #     xtrain=data[k]['train']['x']
    #     ytrain=data[k]['train']['y']
    #     xvalid=data[k]['valid']['x']
    #     yvalid=data[k]['valid']['y']
    #     xtest =data[k]['test']['x']
    #     ytest =data[k]['test']['y']
    #     task_list.append(k)
    #     # Test
    #     print ('-'*40)
    #     test_loss, test_acc, comp_ratio = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
    #     print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

    #     # save accuracy
    #     jj = 0
    #     for ii in np.array(task_list)[0:task_id+1]:
    #         if jj < task_id:
    #             test_acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
    #         else:
    #             xtest = data[ii]['test']['x']
    #             ytest = data[ii]['test']['y']

    #             _, test_acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test", comp_flag=False)
    #             mask_comp_ratio.append(comp_ratio/32)
    #         jj +=1
    #     print('Accuracies =')
    #     for i_a in range(task_id+1):
    #         print('\t',end='')
    #         for j_a in range(i_a + 1):
    #             print('{:5.1f} '.format(test_acc_matrix[i_a,j_a]),end='')
    #         print('')

    #     # update task id
    #     task_id +=1
    # print(new_dist_list)
    # for i in range(9):
    #     dist_tensor[0,i] = new_dist_list[i]
    # dist_tensor_mean = torch.mean(dist_tensor)
    # dist_tensor_var = torch.var(dist_tensor)
    # dist_tensor = torch.abs((dist_tensor - dist_tensor_mean) / dist_tensor_var)
    # print('-'*50)

    # # Simulation Results
    # safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".comp_ratio", np.array(mask_comp_ratio))
    # safe_save("results_reinit/csnb_cifar100_10/" + save_name + ".test_acc", test_acc_matrix)
    # sysparsity_per_layer = print_sparsity(consolidated_masks)
    # all_sparsity = global_sparsity(consolidated_masks)
    # print("Global Sparsity: {}".format(all_sparsity))
    # print("Bit Mask Capacity: {}%".format(np.sum(mask_comp_ratio)))

    # print ('Task Order : {}'.format(np.array(task_list)))
    # print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([test_acc_matrix[i,i] for i in range(len(taskcla))] )))
    # print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(test_acc_matrix[len(taskcla) - 1])))
    # bwt=np.mean((test_acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    # print ('Backward transfer: {:5.2f}%'.format(bwt))
    # print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    # print('-'*50)
    # print(args)

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.7, metavar='SPARSITY',
                        help="Target current sparsity for each layer")
    parser.add_argument('--sparsity_pre', type=float, default=0.3, metavar='SPARSITY',
                        help="spasity for pre mask")
    # Model parameters
    parser.add_argument('--model', type=str, default="alexnet", metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # Deep compression
    parser.add_argument("--deep_comp", type=str, default="", metavar='COMP',
                        help="Deep Compression Model")
    # Pruning threshold
    parser.add_argument("--prune_thresh", type=float, default=0.25, metavar='PRU_TH',
                        help="Pruning threshold for Deep Compression")
    # data parameters
    parser.add_argument('--loader', type=str,
                        default='task_incremental_loader',
                        help='data loader to use')
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')
    parser.add_argument('--data_path', default='./data/', help='path where data is located')
    parser.add_argument("--dataset",
                        default='cifar100_100',
                        type=str,
                        #required=True,
                        choices=['mnist_permutations', 'cifar100_100', 'cifar100_superclass', 'tinyimagenet', 'pmnist'],
                        help="Dataset to train and test on.")
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')

    parser.add_argument("--workers", default=4, type=int, help="Number of workers preprocessing the data.")

    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")
    parser.add_argument("--class_order", default="random", type=str, choices=["random", "chrono", "old", "super"],
                        help="define classes order of increment ")
    # For cifar100
    parser.add_argument('--n_tasks', type=int, default=50,
                        help='total number of tasks, invalid for cifar100_superclass')
    parser.add_argument('--shuffle_task', default=False, action='store_true',
                        help='Invalid for cifar100_superclass')

    parser.add_argument('--encoding', type=str, default="huffman", metavar='',
                        help="")
    parser.add_argument('--gpm_thro', type=float, default=0.95, metavar='gradient projection',
                        help='gpm_thro')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')
    parser.add_argument('--savename', type=str, default='./logs/CIFAR100/',
                        help='save path')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))
    args.sparsity = 1 - args.sparsity
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)