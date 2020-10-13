import os
import numpy as np

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import BertCollator
from transformers import *
from slp.data.bertamz import AmazonZiser17
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.classifier import BertClassifier
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import BertTrainer
from slp.util.embeddings import EmbeddingsLoader
from slp.util.parallel import DataParallelCriterion, DataParallelModel

import argparse
parser = argparse.ArgumentParser(description="Domains and losses")
parser.add_argument("-s", "--source", default="books", help="Source Domain")
parser.add_argument("-t", "--target", default="dvd", help="Target Domain")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target
#targets = ["dvd", "books", "electronics", "kitchen"]

def transform_pred_tar(output):
    y_pred, targets, d  = output
    return y_pred, targets

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
            #import ipdb; ipdb.set_trace()
            pred = trainer.model(inputs)[0]
            metric.update((pred, label))
    acc = metric.compute()
    return acc

#DEVICE = 'cpu'
#DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

collate_fn = BertCollator(device='cpu')

if __name__ == '__main__':
    dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True, cldata=False)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    perm = torch.randperm(len(indices))
    val_size = 0.2
    val_split = int(np.floor(val_size * dataset_size))
    train_indices = perm[val_split:]
    val_indices = perm[:val_split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)

    if TARGET == "books":
       pre = './obooks'
    elif TARGET == "dvd":
       pre = './odvd'
    elif TARGET == "electronics":
       pre = './oele'
    else:
       pre = './okit'

    #config = BertConfig.from_json_file('./config.json')
    #bertmodel = BertModel.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(pre)
    #for names, parameters in model.bert.named_parameters():
    #    parameters.requiers_grad=False

    #optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy()
    }
    path = SOURCE + TARGET
    trainer = BertTrainer(model, optimizer,
                      newbob_period=3,
                      checkpoint_dir=os.path.join('./checkpoints/out/bertpre/', path),
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=3,
                      loss_fn=criterion,
                      device=DEVICE,
                      parallel=False)
    #trainer.fit(train_loader, val_loader, epochs=10)
    trainer = BertTrainer(model, optimizer=None,
                      checkpoint_dir=os.path.join('./checkpoints/out/bertpre/', path),
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)
    #for TARGET in targets:
    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=True, cldata=False)
    test_loader = DataLoader(
         dataset2,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fn)
    file = "PT"+ path + ".txt"
    with open(file, "w") as f:
        print(SOURCE, file=f)
        print(TARGET, file=f)
        print(evaluation(trainer, test_loader, DEVICE), file=f)

