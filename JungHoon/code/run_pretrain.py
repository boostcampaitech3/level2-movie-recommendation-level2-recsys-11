import wandb
# 여기에 project명 쓰고 entity에는 자기 W&B 아이디 쓰면 됨.
# 그리고 이거 파일 돌리기 이전에 터미널 창에 "wandb login" 치고 로그인 해줘야 함.
# 로그인하는 방법은 링크에 나와 있음.
# https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
wandb.init(project="MovieRec", entity="iksadnorth")

import copy
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import PretrainDataset
import datasets
from models import S3RecModel
import models
from trainers import PretrainTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
)

from util import LoadJson


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Pretrain", type=str)

    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument(
        "--pre_epochs", type=int, default=300, help="number of pre_train epochs"
    )
    parser.add_argument("--pre_batch_size", type=int, default=512)

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    LoadJson.dump(args, "/opt/ml/baseline/code/config.json")
    PreDataset = getattr(datasets, args.PreDataset)
    Model = getattr(models, args.Model)

    set_seed(args.seed)
    check_path(args.output_dir)

    args.checkpoint_path = os.path.join(args.output_dir, f"Pretrain_{args.data_name}_{args.output_name}.pt")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + args.data_name + '.txt'
    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute

    model = Model(args=args)
    trainer = PretrainTrainer(model, None, None, None, None, args)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    
    # wandb config 설정.
    wandb_config = copy.deepcopy(args.__dict__)
    wandb_config.pop('item2attribute')
    wandb.config.update(wandb_config)
    
    # wandb run 이름 설정.
    RUN_NAME=f"{args.model_name}_{args.data_name}_{args.output_name}_{str(id(args))[-4:]}"
    wandb.run.name = RUN_NAME

    for epoch in range(args.pre_epochs):

        pretrain_dataset = PreDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size
        )

        losses = trainer.pretrain(epoch, pretrain_dataloader)

        ## comparing `sp_loss_avg``
        early_stopping(np.array([-losses["sp_loss_avg"]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        losses.pop('epoch')
        wandb.log(losses)


if __name__ == "__main__":
    main()
