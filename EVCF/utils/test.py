from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np

import time

import os
import bottleneck as bn
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def test_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    
   # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0


    # set model to evaluation mode
    model.eval()

    # evaluate
    
    predictions = []
    first=True

    for batch_idx, (train, test) in enumerate(train_loader):
        if args.cuda:
            train, test = train.cuda(), test.cuda()
        train, test = Variable(train), Variable(test) #! volatile deprecated

        x = train
        print(batch_idx)
        with torch.no_grad():
            # calculate loss function
            loss, RE, KL = model.calculate_loss(x, average=True)

            evaluate_loss += loss.data.item()
            evaluate_re += -RE.data.item()
            evaluate_kl += KL.data.item()

            # Calculate NDCG & Recall
            pred_val = model.reconstruct_x(x).cpu().detach().numpy()
            if batch_idx <1:
                # print(f"pred_val : {pred_val}\n")
                print(f"array pred_val.size :{pred_val.shape}\n")
                # print(f"pred_val[0] :{pred_val[0]}\n")
                print(f"array pred_val.size :{pred_val[0].shape}\n")
                batch_users = pred_val.shape[0]
                print(f'batch_users : {batch_users}')
                idx_topk_part = bn.argpartition(-pred_val, 30, axis=1)
                print(f'idx_topk part: {idx_topk_part}')
                topk_part = pred_val[np.arange(batch_users)[:, np.newaxis],
                                idx_topk_part[:, :30]]
                print(f'topk_part : {topk_part}')
                idx_part = np.argsort(-topk_part, axis=1)
                print(f'idx_part : {idx_part}')
            if first:
                predictions=pred_val
                first = False
            else:
                predictions = np.concatenate([predictions,pred_val],axis=0)
            ###
            if batch_idx <1:
                # print(f"pred_val : {pred_val}\n")
                print(f"pred_val.size :{pred_val.shape}\n")
                print(f'user0 : {pred_val[0]}\n')
                
                # print(f"pred_val[0] :{pred_val[0]}\n")
                print(f"pred_val.size :{pred_val[0].shape}\n")

    return predictions
                       
           
