from __future__ import print_function
import argparse

import torch
import torch.optim as optim

from utils.optimizer import AdamNormGrad

import os
import numpy as np
import pandas as pd
import datetime

from utils.load_data import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description='VAE+VampPrior')
# arguments for optimization
parser.add_argument('--batch_size', type=int, default=200, metavar='BStrain',
                    help='input batch size for training (default: 200)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='BStest',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='E',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warm-up')
parser.add_argument('--max_beta', type=float, default=1., metavar='B',
                    help='maximum value of beta for training')
parser.add_argument('--inference', action='store_true', default=False, 
                    help='using inference')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

# model: latent size, input_size, so on
parser.add_argument('--num_layers', type=int, default=1, metavar='NL',
                    help='number of layers')

parser.add_argument('--z1_size', type=int, default=200, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=200, metavar='M2',
                    help='latent size')
parser.add_argument('--hidden_size', type=int, default=600 , metavar="H",
                    help='the width of hidden layers')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=1000, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='hvamp', metavar='MN',
                    help='model name: baseline, vamp, hvamp, hvamp1')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous, multinomial')

parser.add_argument('--gated', action='store_true', default=False,
                    help='use gating mechanism')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='preprocessed_data', metavar='DN',
                    help='name of the dataset:  ml20m, netflix, pinterest, preprocessed_data')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

# note
parser.add_argument('--note', type=str, default="none", metavar='NT',
                    help='additional note on the experiment')
parser.add_argument('--no_log', action='store_true', default=False,
                    help='print log to log_dir')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}  #! Changed num_workers: 1->0 because of error

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:16]

    model_name = args.dataset_name + '_' + args.model_name + '_' + \
                 '(K_' + str(args.number_components) + ')' + \
                 '_' + args.input_type + '_beta(' + str(args.max_beta) + ')' + \
                 '_layers(' + str(args.num_layers) + ')' + '_hidden(' + str(args.hidden_size) + ')' + \
                 '_z1(' + str(args.z1_size) + ')' + '_z2(' + str(args.z2_size) + ')'

    # DIRECTORY FOR SAVING
    snapshots_path = 'snapshots/'
    dir = snapshots_path + args.model_signature + '_' + model_name + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    train_loader, val_loader, test_loader, all_loader, args = load_dataset(args, **kwargs)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'baseline':
        from models.Baseline import VAE
    elif args.model_name == 'vamp':
        from models.Vamp import VAE
    elif args.model_name == 'hvamp':
        from models.HVamp import VAE
    elif args.model_name == 'hvamp1':
        from models.HVamp_1layer import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    if args.cuda:
        model.cuda()

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # ======================================================================================================================
    print(args)
    log_dir = "vae_experiment_log_" + str(os.getenv("COMPUTERNAME")) +".txt"

    open(log_dir, 'a').close()

    # ======================================================================================================================
    print('perform experiment')
    from utils.perform_experiment import experiment_vae
    experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, log_dir, model_name = args.model_name)
    # ======================================================================================================================

def inference(args, kwargs):
    from utils.test import test_vae as test

    all_predictions = []
    model_name = args.dataset_name + '_' + args.model_name + '_' + \
                 '(K_' + str(args.number_components) + ')' + \
                 '_' + args.input_type + '_beta(' + str(args.max_beta) + ')' + \
                 '_layers(' + str(args.num_layers) + ')' + '_hidden(' + str(args.hidden_size) + ')' + \
                 '_z1(' + str(args.z1_size) + ')' + '_z2(' + str(args.z2_size) + ')'

    # DIRECTORY FOR SAVING
    snapshots_path = '/opt/ml/input/EVCF/snapshots/'
    dir = snapshots_path + args.model_signature + '_' + model_name + '/'
    print(" model directory : ", dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_loader, val_loader, test_loader, all_loader, args = load_dataset(args, **kwargs)
    test_model = torch.load(dir + args.model_name + '.model')
    #test_model = torch.load('/opt/ml/input/EVCF/snapshots/2022-04-07_preprocessed_data_hvamp_(K_1000)_binary_beta(0.3)_layers(2)_hidden(600)_z1(200)_z2(200)/hvamp.model')

    print(dir + args.model_name + '.model')
    all_predictions = test(args, test_model, all_loader, test_loader, 9999, dir, mode='test')
    print(all_predictions.shape)
    

    temp=pd.DataFrame(all_predictions)

    unique_sid=pd.read_csv('/opt/ml/input/EVCF/data/train/Preprocessed/unique_sid.txt',sep=" ",header=None)

    unique_uid=pd.read_csv('/opt/ml/input/EVCF/data/train/Preprocessed/unique_uid.txt',sep=" ",header=None)

    id2show = dict((i, sid) for (i, sid) in enumerate(unique_sid.squeeze()))
    id2profile = dict((i, pid) for (i, pid) in enumerate(unique_uid.squeeze()))

    column=list(temp.columns)
    origin_mid=[id2show[x] for x in column]

    row=list(temp.index)
    origin_uid=[id2profile[x] for x in row]

    temp.columns=origin_mid
    temp.index=origin_uid

    raw_data=pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')

    watchedm=raw_data.groupby('user')['item'].apply(list)


    from tqdm import tqdm
    sumbission=dict()
    sumbission={'user': [],'item': []}
    sumbission

    print('temp[0] : ' , temp.iloc[0])
    i=0
    for row in tqdm(temp.iterrows(),total=31360):
        userid=row[0]
        movies=row[1]
        watchedmovies=watchedm[userid]
        if i<1:
            print(f'userid : {userid} / movies : {movies} / watchedmovies : {watchedmovies}')
        for _ in range(10):
            sumbission['user'].append(userid)
        
        itemp=[]
        for movie in reversed(list(movies.sort_values().index)):
            if i<1:
                print(list(reversed(list(movies.sort_values().index)))[:20])
            if len(itemp)==10:
                break
            else:
                if movie not in watchedmovies:
                    itemp.append(movie)
                    if i<1:
                        print(f'movie : {movie} \n itemp : {itemp}')

        sumbission['item']+=itemp
        if i<1:
            i+=1

    sumbission=pd.DataFrame(sumbission)
    sumbission = sumbission.sort_values('user')
    #sumbission = sumbission.drop(['Unnamed: 0'],axis='columns')
    #sumbission.index = [x for x in range(313600)]
    sumbission.to_csv('0408_1_testing.csv')

    


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    run(args, kwargs)
    if args.inference:
        inference(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
