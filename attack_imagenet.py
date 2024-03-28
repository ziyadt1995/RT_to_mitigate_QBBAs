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
    2. set the black-box attack on ImageNet
    3. set the device, model, criterion, training schedule
    4. start the attack process and get labels
    5. save attack result
    
'''

"""
Script for running black-box attacks
"""


import json
import math
import os
import time
import sys
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import numpy as np
import pandas as pd
#import tensorflow as tf
import torch

from datasets.dataset import Dataset
from utils.compute import tf_nsign, sign
from utils.misc import config_path_join, src_path_join, create_dir, get_dataset_shape
#from utils.model_loader import load_torch_models, load_torch_models_imagesub

from attacks.score.sign_attack import SignAttack
from attacks.score.square_attack import SquareAttack
from attacks.decision.sign_opt_attack import SignOPTAttack
from attacks.decision.rays_attack import RaySAttack
#from attacks.score.score_black_box_attack import ScoreBlackBoxAttack



if (1):
    config = "../RT_to_mitigate_QBBAs/config-jsons/imagenet_square_linf_config.json" #os.sys.argv[1]
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

        dset = Dataset(config['dset_name'], config)
        epsilon = config['attack_config']['epsilon']
        
        from robustness.datasets import CIFAR, ImageNet
        from robustness.model_utils import make_and_restore_model
        import numpy as np
        ds = ImageNet('/path/to/cifar')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                     resume_path='../imagenet/imagenet_linf_4.pt')
        model.eval()
        
        """
        model_name = config['modeln']
        if config['dset_name'] == 'imagenet_sub':
            model = load_torch_models_imagesub(model_name)
        else:
            model = load_torch_models(model_name)
        """

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
            lb=dset.min_value,
            ub=dset.max_value
        )

        target = config["target"]
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import os
        #from adversarial_robustness import model_zoo
        import torch
        import torchvision
        from tqdm import tqdm
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        no_cuda=False
        use_cuda = not no_cuda and torch.cuda.is_available()
        seed=1
        torch.manual_seed(seed)
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
        torch.backends.cudnn.benchmark = True
   
        with torch.no_grad():

            # Iterate over the samples batch-by-batch
            num_eval_examples = config['num_eval_examples']
            eval_batch_size = config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()
            bat=0
            k=0

            for ibatch in range(num_batches):
                
                #if(k<3):
                #    k=k+1
                #    continue
               
                bstart = ibatch * eval_batch_size
                if(bstart==1000):
                    break
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = dset.get_eval_data(bstart, bend)
                print("bstart:",bstart)
                print("bend:",bend)
                print("x_batch: ",x_batch.shape)
                print("y_batch: ",y_batch.shape)
                #x_batch=x_batch[bstart:bend]
                #y_batch=y_batch[bstart:bend]
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2)).cuda()
                
                x_batch = torch.FloatTensor(x_batch).cuda()
                y_batch = torch.LongTensor(y_batch).cuda()
                
                
                #x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                

                if target:
                    def get_label(target_type):
                        _, logit = model(torch.FloatTensor(x_batch.transpose(0,3,1,2)))
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
                    y_batch = get_label(config["target_type"])

                if config['attack_name'] in ["SignOPTAttack","HSJAttack","GeoDAttack","OptAttack","EvolutionaryAttack",
                                            "SignFlipAttack","RaySAttack","BoundaryAttack"]:
                    logs_dict = attacker.run(x_batch, y_batch, model, target, dset)

                else:
                    def loss_fct(xs, es = False):
                        
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        
                        #print(x_eval[0])
                        #print(x_ori[0])
                        #print(x_eval.shape)
                        #print("loss_fct")
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        #x_eval = x_eval + 0.05 * torch.randn_like(x_eval)
                        
                        y_logit = model(x_eval)[0]
                        #print("Ylogit shape:",y_logit[0].shape)
                        correct = torch.argmax(y_logit, dim=1)
                        y_logit01=torch.zeros(y_logit.shape).scatter(1,correct.unsqueeze(1),1.0)
                        
                        for i in range(10):
                            a=model(x_eval)[0]
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
                        #print(xs.shape)
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
                        #print(x_eval.shape)
                        #print("earlystopcritfct")
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                    #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        #x_eval = x_eval + 0.05 * torch.randn_like(x_eval)
                        
                        """
                        y_logit = model(x_eval)
                        for i in range(10):
                            y_logit = y_logit + model(x_eval)
                        y_logit=y_logit[0]/11
                        """
                        
                        #To use RND as a defense, use model prediction as model(x_eval + 0.05 * torch.randn_like(x_eval))
                        
                        y_logit=model(x_eval)
                     
                        correct = torch.argmax(y_logit[0], dim=1)
                        y_logit=torch.zeros(y_logit[0].shape).scatter(1,correct.unsqueeze(1),1.0)
                        for i in range(10):
                            y_logitt=model(x_eval)
                            correct = torch.argmax(y_logitt[0], dim=1)
                            y_logitt=torch.zeros(y_logitt[0].shape).scatter(1,correct.unsqueeze(1),1.0)
                            y_logit=y_logit+y_logitt
                  
                        #y_logit=y_logit[0]
                            
                        
                        
                        #print(y_logit[0])
                        #y_logit = y_logit.detach()
                        
                        correct = torch.argmax(y_logit, dim=1) == y_batch
                        
                        
                        if target:
                            return correct
                        else:
                            y_pred = model(x_eval)
                            correctt = (torch.argmax(y_pred[0],axis=1) == y_batch).sum().item()
                            print("correct:",correctt)
                            return ~correct
                        
                    

                    logs_dict = attacker.run(x_batch, loss_fct, early_stop_crit_fct, bat)
                    
                    
                print(attacker.result())
        bat=bat+1

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
