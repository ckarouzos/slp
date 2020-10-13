import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from ignite.metrics import Accuracy, Loss
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              SubsetRandomSampler)
from transformers import *

from slp.data.bertamz import AmazonZiser17, NewLabelsData, MyConcat
from slp.data.collators import BertCollator
from slp.data.transforms import SpacyTokenizer, ToTensor, ToTokenIds
from slp.modules.classifier import BertClassifier
from slp.modules.doublebert import *
from slp.modules.doublebert import (DoubleBertCollator, DoubleHeadBert,
                                    DoubleLoss)
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import (AugmentBertTrainer, BertTrainer,
                                 DoubleBertTrainer)
from slp.util.embeddings import EmbeddingsLoader
from slp.util.parallel import DataParallelCriterion, DataParallelModel

parser = argparse.ArgumentParser(description="Domains and losses")
parser.add_argument("-s", "--source", default="books", help="Source Domain")
parser.add_argument("-t", "--target", default="dvd", help="Target Domain")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target
#targets = ["dvd", "books", "electronics", "kitchen"]

def transform_pred_tar(output):
    y_pred, targets, d  = output
    doms = d['domains']
    #if not doms[0]:

    return y_pred, targets
    #else:
    #   import ipdb; ipdb.set_trace()

def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']

    return d_pred, d_targets

def evaluation(trainer, test_loader, device):
    trainer.model.eval()
    predictions = []
    labels = []
    metric = Accuracy()
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            inputs = batch[0].to(device)
            label = batch[1].to(device)
            domain = batch[2].to(device)
            #import ipdb; ipdb.set_trace()
            pred = trainer.model(inputs, source=domain[0])[0]
            metric.update((pred, label))
    acc = metric.compute()

    return acc

#DEVICE = 'cpu'
#DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

collate_fA = BertDCollator(device='cpu')
collate_fB = BertLMCollator(tokenizer=BertTokenizer('./okit/vocab.txt', do_lower_case=True))
collate_fn = DoubleBertCollator(collate_fA, collate_fB)


def split_dataset(dataset, val_size=.2):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[int(val_size * len(dataset)):]
    val_indices = indices[:int(val_size * len(dataset))]

    return train_indices, val_indices


def dataloaders_from_datasets(source_dataset, target_dataset,
                              batch_train, batch_val, circle,
                              val_size=0.2):
    s_train_indices , s_val_indices = split_dataset(source_dataset, val_size=.2)
    t_train_indices , t_val_indices = split_dataset(target_dataset, val_size=.2)

    train_source = Subset(source_dataset, s_train_indices)
    train_target = Subset(target_dataset, t_train_indices)
    val_source = Subset(source_dataset, s_val_indices)
    val_target = Subset(target_dataset, t_val_indices)
    #import ipdb; ipdb.set_trace()
    train_set = ConcatDataset([train_source, train_target])
    val_set = ConcatDataset([val_source, val_target])
    train_source_size = len(train_source)
    val_source_size = len(val_source)

    s_train_indices = list(range(len(train_source)))
    s_val_indices = list(range(len(val_source)))
    t_train_indices = [i + len(train_source) for i in list(range(len(train_target)))]
    t_val_indices = [i + len(val_source) for i in list(range(len(val_target)))]

    # dataset = ConcatDataset([source_dataset, target_dataset])
    # s_dataset_size = len(source_dataset)
    # s_indices = list(range(s_dataset_size))
    # s_val_split = int(np.floor(val_size * s_dataset_size))
    # s_train_indices = s_indices[s_val_split:]
    # s_val_indices = s_indices[:s_val_split]

    # t_dataset_size = len(target_dataset)
    # t_indices = list(range(t_dataset_size))
    # t_val_split = int(np.floor(val_size*t_dataset_size))
    # t_train_indices = t_indices[t_val_split:]
    # t_val_indices = t_indices[:t_val_split]


    train_sampler = DoubleSubsetRandomSampler(
        s_train_indices, t_train_indices, train_source_size,
        batch_train, batch_train * circle
    )
    val_sampler = DoubleSubsetRandomSampler(
        s_val_indices, t_val_indices, val_source_size,
        batch_val, batch_val * circle
    )

    # train_sampler = DoubleSubsetRandomSampler(s_train_indices, t_train_indices, s_dataset_size, batch_train, batch_train*circle)
    #val_sampler = SubsetRandomSampler(s_val_indices)
    # val_sampler = DoubleSubsetRandomSampler(s_val_indices, t_val_indices, s_dataset_size, batch_val, batch_val*circle)


    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_train,
    #     sampler=train_sampler,
    #     drop_last=False,
    #     collate_fn=collate_fn)
    # val_loader = DataLoader(
    #     source_dataset,
    #     batch_size=batch_val,
    #     sampler=val_sampler,
    #     drop_last=False,
    #     collate_fn=collate_fn)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_val,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn)

    return train_loader, val_loader

if __name__ == '__main__':
    dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True, cldata=False)
    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=False, cldata=True)
    train_loader, val_loader = dataloaders_from_datasets(dataset, dataset2,
                                                         4, 4, 8)

    if TARGET == "books":
       pre = './obooks'
    elif TARGET == "dvd":
       pre = './odvd'
    elif TARGET == "electronics":
       pre = './oele'
    else:
       pre = './okit'

    model = DoubleHeadBert.from_pretrained(pre)
    #for names, parameters in model.bert.named_parameters():
    #    parameters.requiers_grad=False

    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    in_fn = nn.CrossEntropyLoss()
    criterion = DoubleLoss(in_fn)
    metrics = {
        'loss': Loss(criterion)
        #'accuracy': Accuracy(transform_pred_tar)
    }
    path=SOURCE+TARGET
    trainer = DoubleBertTrainer(model, optimizer,
                      newbob_period=3,
                      checkpoint_dir=os.path.join('./checkpoints/out/doublem', path),
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=3,
                      validate_every=1,
                      accumulation_steps=5,
                      loss_fn=criterion,
                      device=DEVICE,
                      parallel=False)

    trainer.fit(train_loader, val_loader, epochs=10)
    trainer = DoubleBertTrainer(model, optimizer=None,
                      checkpoint_dir=os.path.join('./checkpoints/out/doublem', path),
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)

    dataset3 = AmazonZiser17(ds=TARGET, dl=0, labeled=True, cldata=False)

    final_test_loader = DataLoader(
         dataset3,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fA)
    file="m"+path+".txt"
    with open(file, "w") as f:
       print(SOURCE, file=f)
       print(TARGET, file=f)
       print(evaluation(trainer, final_test_loader, DEVICE), file=f)
