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
parser.add_argument("-dir", "--directory", default="./checkpoints/out/double")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target
directory = args.directory
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

def make_tsne(trainer, train_loader, test_loader, device, path):
    trainer.model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for index, batch in enumerate(train_loader):
            inputs = batch[0].to(device)
            if batch[1].to(device)==0:
                labels.append(0)
            else:
                labels.append(1)
            pred = trainer.model.bert(inputs)[1]
            predictions.append(pred)

        for index, batch in enumerate(test_loader):
            inputs = batch[0].to(device)
            if batch[1].to(device)==0:
                labels.append(2)
            else:
                labels.append(3)
            pred = trainer.model.bert(inputs)[1]
            predictions.append(pred)
        arr = np.array([i.cpu().numpy() for i in predictions])
        lab = np.array([i for i in labels])
        arr = np.reshape(arr, (4000, 768))
        lab = np.reshape(lab, (4000, 1))
        #import ipdb; ipdb.set_trace()
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        #import umap
        #X = umap.UMAP(n_components=2).fit_transform(arr)
        X = TSNE(n_components=2).fit_transform(arr)
        fig, ax = plt.subplots(figsize=(10,8))
        lab = np.array([l.item() for l in lab])
        Xx1 = np.array([x for (x,y),l in zip(X[:2000],lab[:2000]) if l==0])
        Xy1 = np.array([y for (x,y),l in zip(X[:2000],lab[:2000]) if l==0])
        Xx2 = np.array([x for (x,y),l in zip(X[:2000],lab[:2000]) if l!=0])
        Xy2 = np.array([y for (x,y),l in zip(X[:2000],lab[:2000]) if l!=0])
        Xx3 = np.array([x for (x,y),l in zip(X[2000:],lab[2000:]) if l==2])
        Xy3 = np.array([y for (x,y),l in zip(X[2000:],lab[2000:]) if l==2])
        Xx4 = np.array([x for (x,y),l in zip(X[2000:],lab[2000:]) if l!=2])
        Xy4 = np.array([y for (x,y),l in zip(X[2000:],lab[2000:]) if l!=2])
        ax.scatter(Xx1, Xy1, s=5, c='r', marker='o', label='Source Negative')
        ax.scatter(Xx2, Xy2, s=5, c='g', marker='o', label='Source Positive')
        ax.scatter(Xx3, Xy3, s=5, c='y', marker='p', label='Target Negative')
        ax.scatter(Xx4, Xy4, s=5, c='b', marker='*', label='Target Positive')
        ax.legend()
        f = path + '.svg'
        fig.savefig(fname=f)


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

    #bertmodel = BertModel.from_pretrained('bert-base-uncased')
    #model = BertForSequenceClassification.from_pretrained('./okit')
    from slp.modules.doublebert import DoubleHeadBert
    from slp.trainer.trainer import DoubleBertTrainer
    model = DoubleHeadBert.from_pretrained('./okit')
    #optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy()
    }
    #import pdb; pdb.set_trace()
    #trainer = DoubleBertTrainer(model, optimizer,
    #                  newbob_period=3,
    #                  checkpoint_dir=directory,
    #                  metrics=metrics,
    #                  non_blocking=True,
    #                  retain_graph=True,
    #                  patience=3,
    #                  loss_fn=criterion,
    #                  device=DEVICE,
    #                  parallel=False)
    #trainer.fit(train_loader, val_loader, epochs=10)
    path=SOURCE+TARGET
    trainer = DoubleBertTrainer(model, optimizer=None,
                      checkpoint_dir=os.path.join(directory,path),
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)
    #dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=True, cldata=False)
    #import ipdb; ipdb.set_trace()
    #for TARGET in targets:
    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=True, cldata=False)
    train_loader = DataLoader(
         dataset,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fn)
    test_loader = DataLoader(
         dataset2,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fn)
    make_tsne(trainer, train_loader, test_loader, DEVICE, path)
    #print(SOURCE)
    #    print(TARGET)
    #    print(evaluation(trainer, test_loader, DEVICE))
