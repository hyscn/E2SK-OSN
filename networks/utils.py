# Authorized by Haeyong Kang.

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse,time
import torchvision
from torchvision import datasets, transforms
import numpy as np
from copy import deepcopy
import os
import os.path
import math
from itertools import combinations, permutations

# ------------------prime generation and prime mod tables---------------------
# Find closest number in a list

def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)

class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()

    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]
        x_ori_perm = x_ori[index, :]
        lam = lam.view((N, 1, 1, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b

class AugModule_mnist(nn.Module):
    def __init__(self):
        super(AugModule_mnist, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]

        x_ori_perm = x_ori[index, :]

        lam = lam.view((N, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def save_model_params(saver_dict, model, task_id):

    print ('saving model parameters ---')

    saver_dict[task_id]['model']={}
    for k_t, (m, param) in enumerate(model.named_parameters()):
        saver_dict[task_id]['model'][m] = param
        print (k_t,m,param.shape)
    print ('-'*30)

    return saver_dict

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor

def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(num_primes):
    primes = []
    for num in range(2,np.inf):
        if is_prime(num):
            primes.append(num)
            print(primes)

        if len(primes) >= num_primes:
            return primes

def checker(per_task_masks, consolidated_masks, task_id):
    # === checker ===
    for key in per_task_masks[task_id].keys():
        # Skip output head from other tasks
        # Also don't consolidate output head mask after training on new tasks; continue
        if "last" in key:
            if key in curr_head_keys:
                consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
            continue

        # Or operation on sparsity
        if 'weight' in key:
            num_cons = consolidated_masks[key].sum()
            num_prime = (prime_masks[key] > 0).sum()

            if num_cons != num_prime:
                print('diff.')

def print_sparsity(consolidated_masks, percent=1.0, item=False):
    sparsity_dict = {}
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            sparsity = torch.sum(mask == 1) / np.prod(mask.shape)
            print("{:>12} {:>2.4f}".format(key, sparsity ))

            if item :
                sparsity_dict[key] = sparsity.item() * percent
            else:
                sparsity_dict[key] = sparsity * percent

    return sparsity_dict

def global_sparsity(consolidated_masks):
    denum, num = 0, 0
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            num += torch.sum(mask == 1).item()
            denum += np.prod(mask.shape)

    return num / denum

# def get_representation_matrix (net,device,x,y=None,task_id=0,mask=None,mode='valid',random=True):
#     # Collect activations by forward pass
#     r=np.arange(x.size(0))

#     if random:
#         np.random.shuffle(r)
#         r=torch.LongTensor(r).to(device)
#         b=r[0:300] # Take random training samples
#         batch_list=[300,300,300]
#     else:
#         r=torch.LongTensor(r).to(device)
#         b=r # Take all valid samples
#         batch_list=[r.size(0),r.size(0),r.size(0)]

#     example_data = x[b].view(-1,28*28)
#     example_data = example_data.to(device)
#     example_out  = net(example_data, task_id, mask, mode)

#     mat_list=[] # list contains representation matrix of each layer
#     act_key=list(net.act.keys())

#     for i in range(len(act_key)):
#         bsz=batch_list[i]
#         act = net.act[act_key[i]].detach().cpu().numpy()
#         activation = act[0:bsz].transpose()
#         mat_list.append(activation)

#     print('-'*30)
#     print('Representation Matrix')
#     print('-'*30)
#     for i in range(len(mat_list)):
#         print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
#     print('-'*30)
#     return mat_list

def get_representation (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data = example_data.to(device)
    #example_out  = net(example_data)
    
    batch_list=[300,300,300] 
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list   

def get_feature_matrix(model):
    grad_shape = []
    for k, (m,params) in enumerate(model.named_parameters()):
        if k<15 and len(params.size())!=1:
            grad_shape.append(params.data.shape)
    return grad_shape       


def get_representation_matrix_super_cifar100 (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b]
    example_data = example_data.to(device)
    #example_out  = net(example_data)
    
    batch_list=[2*12,100,125,125] 
    pad = 2
    p1d = (2, 2, 2, 2)
    mat_list=[]
    act_key=list(net.act.keys())
    # pdb.set_trace()
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<2:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i],1,pad)
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = F.pad(net.act[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()
         
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) #?
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list    

def get_representation_matrix_ResNet18_5data(net, device, x, y=None): 
    # Collect activations by forward pass
    net.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100] # ns=100 examples 
    example_data = x[b]
    example_data = example_data.to(device)
    #example_out  = net(example_data)
    
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    #batch_list  = [4,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100]
    batch_list =  [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    # network arch 
    stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 

    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[] # list containing GPM Matrices 
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3 
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat) 

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    print('-'*30)
    return mat_final    


def get_representation_matrix_tiny(net ,device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:50] # Take 125 random samples 
    example_data = x[b]
    example_data = example_data.to(device)
    #example_out  = net(example_data,task_id,consolidated_masks)
    
    batch_list=[6,25,25,25,25] 
    mat_list=[]
    act_key=list(net.act.keys())
    #print(act_key)
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:
            ksz= net.ksize[i]
            print(ksz)
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            print(act.shape)
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    # for i in range(len(mat_list)):
    #     mat_sz = mat_list[i].size(0)
    #     mat_list[i] = 0.01 * mat_list[i] + np.eye(mat_sz)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list


def get_representation_matrix (net ,device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b]
    example_data = example_data.to(device)
    #example_out  = net(example_data,task_id,consolidated_masks)
    
    batch_list=[6,8,8,8,8] 
    mat_list=[]
    act_key=list(net.act.keys())
    #print(act_key)
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range( s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    # for i in range(len(mat_list)):
    #     mat_sz = mat_list[i].size(0)
    #     mat_list[i] = 0.01 * mat_list[i] + np.eye(mat_sz)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list


def update_SGP (args, model, mat_list, threshold, task_id, feature_list=[], importance_list=[]):
    print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-1)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            # update GPM
            feature_list.append(U[:,0:r])
            # update importance (Eq-2)
            importance = ((args.scale_coff+1)*S[0:r])/(args.scale_coff*S[0:r] + max(S[0:r])) 
            importance_list.append(importance)
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-4)
            act_proj = np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            r_old = feature_list[i].shape[1] # old GPM bases 
            Uc,Sc,Vhc = np.linalg.svd(act_proj, full_matrices=False)
            importance_new_on_old = np.dot(np.dot(feature_list[i].transpose(),Uc[:,0:r_old])**2, Sc[0:r_old]**2) ## r_old no of elm s**2 fmt
            importance_new_on_old = np.sqrt(importance_new_on_old)
            
            act_hat = activation - act_proj
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-5)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                # update importances 
                importance = importance_new_on_old
                importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance)) 
                importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
                importance_list[i] = importance # update importance
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            # update importance 
            importance = np.hstack((importance_new_on_old,S[0:r]))
            importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance))         
            importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1) 

            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
                importance_list[i] = importance[0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
                importance_list[i] = importance

    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list, importance_list 


def update_GPM (model, mat_list, threshold, feature_list=[], task_id=None):
    print ('Threshold: ', threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            print('feature_shape:',activation.shape)
            a_sz = activation.shape[0] 
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)

            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i]) #+1
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            a_sz = activation.shape[0] 
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()

            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            a_sz = act_hat.shape[0]
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            accumulated_sval = (sval_total-sval_hat)/sval_total

            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1))
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i] = Ui[:,0:Ui.shape[0]]
            else: 
                feature_list[i]=Ui

    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list

def mask_projected (mask, feature_mat):

    # mask Projections
    for i, key in enumerate(mask.keys()):
        #for j in range(len(feature_mat)):
        if 'weight' in key:
            mask[key] = mask[key] - torch.mm(mask[key].float(), feature_mat[i])
        else:
            None

    return mask

## Define LeNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
