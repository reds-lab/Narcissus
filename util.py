import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
import torch.nn as nn
from collections import OrderedDict

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def torch_normalization(data):
    new_data = data.clone()
    if data.dim() == 4:
        _range1 = torch.max(data[0,0,:,:]) - torch.min(data[0,0,:,:])
        _range2 = torch.max(data[0,1,:,:]) - torch.min(data[0,1,:,:])
        _range3 = torch.max(data[0,2,:,:]) - torch.min(data[0,2,:,:])
        if _range1 > 0:
            new_data[0,0,:,:] = (data[0,0,:,:] - torch.min(data[0,0,:,:])) / _range1
        if _range2 > 0:
            new_data[0,1,:,:] = (data[0,1,:,:] - torch.min(data[0,1,:,:])) / _range2
        if _range3 > 0:
            new_data[0,2,:,:] = (data[0,2,:,:] - torch.min(data[0,2,:,:])) / _range3
    return new_data

def torch_normalization_inv(data,epsilon):
    new_data = data.clone()
    if data.dim() == 4:
        _range1 = torch.max(data[0,0,:,:]) - torch.min(data[0,0,:,:])
        _range2 = torch.max(data[0,1,:,:]) - torch.min(data[0,1,:,:])
        _range3 = torch.max(data[0,2,:,:]) - torch.min(data[0,2,:,:])
        if _range1 > 0:
            new_data[0,0,:,:] = (data[0,0,:,:] - torch.min(data[0,0,:,:])) / _range1
            new_data[0,0,:,:] = new_data[0,0,:,:]*(epsilon*2)/255
            new_data[0,0,:,:] = new_data[0,0,:,:] - epsilon/255
        if _range2 > 0:
            new_data[0,1,:,:] = (data[0,1,:,:] - torch.min(data[0,1,:,:])) / _range2
            new_data[0,1,:,:] = new_data[0,1,:,:]*(epsilon*2)/255
            new_data[0,1,:,:] = new_data[0,1,:,:] - epsilon/255
        if _range3 > 0:
            new_data[0,2,:,:] = (data[0,2,:,:] - torch.min(data[0,2,:,:])) / _range3
            new_data[0,2,:,:] = new_data[0,2,:,:]*(epsilon*2)/255
            new_data[0,2,:,:] = new_data[0,2,:,:] - epsilon/255
    return new_data

def norm_weight(weights):
    norm = torch.sum(weights)
    if norm != 0:
        normed_weights = weights / norm
    else:
        normed_weights = weights
    return normed_weights

def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
      min ||x - u||_2 s.t. ||u||_1 <= eps
    Inspired by the corresponding numpy version by Adrien Gaidon.
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
    eps: float
      radius of l-1 ball to project onto
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/torch.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == 3:
        v = torch.sign(v) * torch.minimum(abs(v), torch.tensor(xi))
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v

def get_dataset_index(target_path,target_label):

    all_content=os.listdir(target_path)

    lab_count = 0
    pass_file = 0
    target_len = 0
    for content in all_content:
        files_name = os.listdir(target_path+content)
        if lab_count == target_label:
            target_len += len(files_name)
            target_list = list(range(pass_file,pass_file+target_len))
        pass_file += len(files_name)
        lab_count += 1
    non_target_list = list(set(list(range(0,pass_file))) - set(target_list))
    return target_list,non_target_list

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,labels):
        self.dataset = dataset
        self.indices = indices
        labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = labels 
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)
        

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
    
def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if images.dim() == 3:
        noise_now = noise.clone()[0,:,:,:]
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images += m(noise_now)
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = noise.clone()
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[i:i+1] += m(noise_now)
            else:
                images[i:i+1] += noise_now
    return images

class posion_label(Dataset):
    def __init__(self, dataset,indices,target):
        self.dataset = dataset
        self.indices = indices
        self.target = target

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        return (image, self.target)

    def __len__(self):
        return len(self.dataset)

class posion_image(Dataset):
    def __init__(self, dataset,indices,noise):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        if idx in self.indices:
            image = torch.clamp(apply_noise_patch(self.noise,image,mode='add'),-1,1)
        label = self.dataset[idx][1]
        return (image, label)

    def __len__(self):
        return len(self.dataset)
    
class posion_image_label(Dataset):
    def __init__(self, dataset,indices,noise,target):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        image = torch.clamp(apply_noise_patch(self.noise,image,mode='add'),-1,1)
        #label = self.dataset[idx][1]
        return (image, self.target)

    def __len__(self):
        return len(self.indices)
    
def destructive_append(l,i):
    l=l[1:]
    l.append(i)
    return l

class get_labels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)
    
def load_pth(input_model,load_file_path):
    loaded_dict = torch.load(load_file_path)
    new_state_dict = OrderedDict()
    for k, v in loaded_dict.items():
        name = k[7:]
        new_state_dict[name] = v 

    input_model.load_state_dict(new_state_dict)
    input_model = input_model.cuda()
    return input_model

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset):
        self.idataset = target_dataset
        self.odataset = outter_dataset

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)