

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
This file is copied from the following source:
link: https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter/blob/master/src/attacks/blackbox/run.attack.py

@inproceedings{
al-dujaili2020sign,
title={Sign Bits Are All You Need for Black-Box Attacks},
author={Abdullah Al-Dujaili and Una-May O'Reilly},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SygW0TEFwH}
}

The original license is placed at the end of this file.

basic structure for main:
    1. config args, save_path
    2. set the black-box attack on cifar-10
    3. set the device, model, criterion, training schedule
    4. start the attack process and get labels
    5. save attack result
    
'''

"""
Script for running black-box attacks
"""

'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import transforms

import random

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, RND=False, RR=False, RRC=False, rfd=False):
        if(RND==True):
            noise=torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            #noise=noise.numpy()
            noise=torch.cuda.FloatTensor(noise)
            x=torch.clamp(x+(noise*0.05), 0, 1)
        if(RR==True):
            x=transforms.RandomRotation(degrees=(-10,10))(x)
        if(RRC==True):
            x=transforms.RandomResizedCrop(size=(64,64),scale=(0.8,1))(x)
            
        out = self.conv1(x)
        out = self.layer1(out)
        if(rfd==True):
            noise=torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            #noise=noise.numpy()
            noise=torch.cuda.FloatTensor(noise)
            x=torch.clamp(x+(noise*0.05), 0, 1)
        out = self.layer2(out)
        if(rfd==True):
            noise=torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            #noise=noise.numpy()
            noise=torch.cuda.FloatTensor(noise)
            x=torch.clamp(x+(noise*0.05), 0, 1)
        out = self.layer3(out)
        if(rfd==True):
            noise=torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            #noise=noise.numpy()
            noise=torch.cuda.FloatTensor(noise)
            x=torch.clamp(x+(noise*0.05), 0, 1)
        out = self.layer4(out)
        if(rfd==True):
            noise=torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            #noise=noise.numpy()
            noise=torch.cuda.FloatTensor(noise)
            x=torch.clamp(x+(noise*0.05), 0, 1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    #print(y.size())

# test()


import json
import math
import os
import time
import sys
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import numpy as np
import pandas as pd
import tensorflow as tf
# import torch as ch
import torch

from datasets.dataset import Dataset
from utils.compute import tf_nsign, sign
from utils.misc import config_path_join, src_path_join, create_dir, get_dataset_shape
#from utils.model_loader import load_torch_models, load_torch_models_imagesub
from utils.compute import tf_nsign, sign, linf_proj_maker, l2_proj_maker

#from attacks.score.nes_attack import NESAttack
#from attacks.score.bandit_attack import BanditAttack
#from attacks.score.zo_sign_sgd_attack import ZOSignSGDAttack
#from attacks.score.sign_attack import SignAttack
#from attacks.score.simple_attack import SimpleAttack
from attacks.score.square_attack import SquareAttack
#from attacks.score.parsimonious_attack import ParsimoniousAttack
#from attacks.score.dpd_attack import DPDAttack

#from attacks.decision.sign_opt_attack import SignOPTAttack
#from attacks.decision.hsja_attack import HSJAttack
#from attacks.decision.geoda_attack import GeoDAttack
#from attacks.decision.opt_attack import OptAttack
#from attacks.decision.evo_attack import EvolutionaryAttack
#from attacks.decision.sign_flip_attack import SignFlipAttack
from attacks.decision.rays_attack import RaySAttack
#from attacks.decision.boundary_attack import BoundaryAttack

if __name__ == '__main__':
    config = "../RT_to_mitigate_QBBAs/config-jsons/cifar10_square_linf_config.json" #os.sys.argv[1]
    exp_id = config.split('/')[-1]
    print("Running Experiment {}".format(exp_id))

    # create/ allocate the result json for tabulation
    data_dir = src_path_join('blackbox_attack_exp')
    create_dir(data_dir)
    res = {}
    cfs = [config]
    print(cfs)


    for _cf in cfs:
        

        #config_file = config_path_join(_cf)
        config_file=_cf
        #tf.reset_default_graph()

        with open(config_file) as config_file:
            config = json.load(config_file)
            
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        print(config)

        #dset = Dataset(config['dset_name'], config)
        
        """
        model_name = config['modeln']
        if config['dset_name'] == 'imagenet':
            model = load_torch_models_imagesub(model_name)
        else:
            model = load_torch_models(model_name)

        print('The Black-Box model: {}'.format(config['modeln']))
        """
        
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        """
        model = PreActResNet18()
     
        model = model.to(device)
        checkpoint = torch.load('model.pth')
        from collections import OrderedDict
        try:
            model.load_state_dict(checkpoint)
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, False)
        """
        model_dir = "resnetadv_trades_tinyimagenet_y=8_2nd_rr4minus4"
        
        model = PreActResNet18().to(device)
        

        model.load_state_dict(torch.load(os.path.join(model_dir, 'model-res-epoch{}.pt'.format(110))))
        model.eval()




        p_norm = config['attack_config']['p']
        print("The attack norm constrain is: {} norm".format(p_norm))
        epsilon = config['attack_config']['epsilon'] / 255.

        # set torch default device:
        #if 'gpu' in config['device'] and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')

        def cw_loss(logit, label, target=False):
            if target:
                # targeted cw loss: logit_t - max_{i\neq t}logit_i
                _, argsort = logit.sort(dim=1, descending=True)
                target_is_max = argsort[:, 0].eq(label)
                second_max_index = target_is_max.long() * argsort[:, 1] + (~ target_is_max).long() * argsort[:, 0]
                target_logit = logit[torch.arange(logit.shape[0]), label]
                second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
                return target_logit - second_max_logit 
            else:
                # untargeted cw loss: max_{i\neq y}logit_i - logit_y
                _, argsort = logit.sort(dim=1, descending=True)
                gt_is_max = argsort[:, 0].eq(label)
                second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
                gt_logit = logit[torch.arange(logit.shape[0]), label]
                second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
                return second_max_logit - gt_logit

        def xent_loss(logit, label, target=False):
            if not target:
                return torch.nn.CrossEntropyLoss(reduction='none')(logit, label)                
            else:
                return -torch.nn.CrossEntropyLoss(reduction='none')(logit, label)                

        # criterion = torch.nn.CrossEntropyLoss(reduce=False)
        criterion = xent_loss
        #criterion = cw_loss


        attacker = eval(config['attack_name'])(
            **config['attack_config'],
            lb=0,
            ub=255
        )

        print(attacker._config())

        target = config["target"]
   
        with torch.no_grad():

            # Iterate over the samples batch-by-batch
            num_eval_examples = config['num_eval_examples']
            eval_batch_size = config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
            
            # #-------------------------------------------------#
            # correct_list = []
            # for ibatch in range(num_batches):
            #     bstart = ibatch * eval_batch_size
            #     bend = min(bstart + eval_batch_size, num_eval_examples)
            #     xx, yy = dset.get_eval_data(bstart, bend)
            #     xx = torch.FloatTensor(xx.transpose(0,3,1,2) / 255.).cuda()
            #     y_batch = torch.LongTensor(yy).cuda()
            #     # to tensor ----------------------------------# 
            #     xx = xx + config['sigma'] * torch.randn_like(xx)
            #     xx = torch.clamp(xx, 0, 1)
            #     _, yy_ = model(xx)
            #     yy_ = yy_.detach()
            #     # correct = torch.argmax(y_logit, axis=1) == y_batch
            #     correct = (torch.argmax(yy_, axis=1) == y_batch).cpu().numpy()
            #     correct_list.append(correct)
            # correct = np.vstack(correct_list)
            # import pdb; pdb.set_trace()
            # print('Noise: {:.5} Clean accuracy: {:.2%}'.format(config['sigma'], np.mean(correct)))
            # #-------------------------------------------------#
            
            
            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()
            bat=0
            k=0

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))
                if(ibatch==1):
                    break

                #x_batch, y_batch = dset.get_eval_data(bstart, bend)
                x_batch=np.load("tinyimagenet_x_val_shuffled500.npy")
                y_batch=np.load("tinyimagenet_y_val_shuffled500.npy")
                x_batch=np.rollaxis(np.clip(x_batch*255,0,255), 1,4)
                print("bstart:",bstart)
                print("bend:",bend)
                print("x_batch: ",x_batch.shape)
                print("y_batch: ",y_batch.shape)
                y_batch = torch.LongTensor(y_batch).cuda()
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2) / 255.).cuda()
                
                if p_norm == 'inf':
                    pass
                else:
                    proj_2 = l2_proj_maker(x_ori, epsilon)

                def get_label(target_type):
                    _, logit = model(torch.FloatTensor(x_batch.transpose(0,3,1,2) / 255.))
                    if target_type == 'random':
                        label = torch.randint(low=0, high=logit.shape[1], size=label.shape).long().cuda()
                    elif target_type == 'least_likely':
                        label = logit.argmin(dim=1) 
                    elif target_type == 'most_likely':
                        label = torch.argsort(logit, dim=1,descending=True)[:,1]
                    elif target_type == 'median':
                        label = torch.argsort(logit, dim=1,descending=True)[:,4]
                    elif 'label' in target_type:
                        label = torch.ones_like(y_batch) * int(target_type[5:])
                    return label.detach()

                if target:
                    y_batch = get_label(config["target_type"])

                if config['attack_name'] in ["SignOPTAttack","HSJAttack","GeoDAttack","OptAttack","EvolutionaryAttack",
                                            "SignFlipAttack","RaySAttack","BoundaryAttack"]:
                    logs_dict = attacker.run(x_batch, y_batch, model, target, dset)

                else:
                    def loss_fct(xs, es = False):
                        if type(xs) is torch.Tensor:
                            x_eval = (xs.permute(0,3,1,2)/ 255.).cuda()
                        else:
                            x_eval = (torch.FloatTensor(xs.transpose(0,3,1,2))/ 255.).cuda()

                        if p_norm == 'inf':
                            x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        else:
                            # proj_2 = l2_proj_maker(x_ori, epsilon)
                            x_eval = proj_2(x_eval)
                            
                        x_eval = torch.clamp(x_eval, 0, 1)    
                        y_logit = model(x_eval.cuda())
                        #print("Ylogit shape:",y_logit[0].shape)
                        correct = torch.argmax(y_logit, dim=1)
                        y_logit01=torch.zeros(y_logit.shape).scatter(1,correct.unsqueeze(1),1.0)
                        
                        for i in range(10):
                            a=model(x_eval.cuda())
                            y_logit=y_logit+a
                            
                            correct = torch.argmax(a, dim=1)
                            y_logitt01=torch.zeros(a.shape).scatter(1,correct.unsqueeze(1),1.0)
                            y_logit01=y_logit01+y_logitt01
                        
                        correct = torch.argmax(y_logit01, dim=1) == y_batch
                        y_logit=y_logit/11
                        
                        
                        #y_logit=y_logit[0]
                        
                        loss = criterion(y_logit, y_batch, target)
                      
                        if es:
                            #y_logit = y_logit.detach()
                            #correct = torch.argmax(y_logit, dim=1) == y_batch
                            if target:
                                return correct, loss.detach()
                            else:
                                return ~correct, loss.detach()
                        else: 
                            return loss.detach()    
                        
                 

                    def early_stop_crit_fct(xs):

                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)/ 255.
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))/ 255.
                        x_eval = torch.clamp(x_eval, 0, 1)
                        
                        
                        
                        
                        y_logit=model(x_eval.cuda())
                     
                        correct = torch.argmax(y_logit, dim=1)
                        y_logit=torch.zeros(y_logit.shape).scatter(1,correct.unsqueeze(1),1.0)
                        for i in range(10):
                            y_logitt=model(x_eval.cuda())
                            correct = torch.argmax(y_logitt, dim=1)
                            y_logitt=torch.zeros(y_logitt.shape).scatter(1,correct.unsqueeze(1),1.0)
                            y_logit=y_logit+y_logitt
                  
                        #y_logit=y_logit[0]
                            
                        
                        
                        #print(y_logit[0])
                        #y_logit = y_logit.detach()
                        
                        correct = torch.argmax(y_logit, dim=1) == y_batch
                        
                        
                        if target:
                            return correct
                        else:
                            y_pred = model(x_eval.cuda())
                            correctt = (torch.argmax(y_pred,axis=1) == y_batch).sum().item()
                            print("correct:",correctt)
                            return ~correct
                        """
                        #---------------------#
                        # sigma = config["sigma"]
                        sigma = 0
                        #---------------------#
                        x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        x_eval = torch.clamp(x_eval, 0, 1)
                        _, y_logit = model(x_eval.cuda())
                        y_logit = y_logit.detach()
                        correct = torch.argmax(y_logit, axis=1) == y_batch
                        # expect_num = 10
                        # y_logit_list = 0
                        # for i in range(expect_num): 
                        #     x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        #     x_eval = torch.clamp(x_eval, 0, 1)
                        #     _, y_logit = model(x_eval.cuda())
                        #     y_logit_list += y_logit.detach()
                        # y_logit_list = y_logit_list/expect_num
                        # correct = torch.argmax(y_logit_list, axis=1) == y_batch
                        
                        if target:
                            return correct
                        else:
                            return ~correct
                        """

                    logs_dict = attacker.run(x_batch, loss_fct, early_stop_crit_fct, bat)
                    
                print(attacker.result())
                print('The Black-Box model: {}'.format(config['modeln']))

        print("Batches done after {} s".format(time.time() - start_time))

        if config['dset_name'] not in res:
            res[config['dset_name']] = [attacker.result()]
        else:
            res[config['dset_name']].append(attacker.result())

        res_fname = os.path.join(data_dir, '{}_res.json'.format(_cf))
        print("Storing tabular data in {}".format(res_fname))
        with open(res_fname, 'w') as f:
            json.dump(res, f, indent=4, sort_keys=True)
'''
    
MIT License
Copyright (c) 2019 Abdullah Al-Dujaili
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
