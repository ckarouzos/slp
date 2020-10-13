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

def alpha_distance(trainer, source_loader, target_loader, device, path):
    trainer.model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for index, batch in enumerate(source_loader):
            inputs = batch[0].to(device)
            pred = trainer.model.bert(inputs)[1]
            predictions.append(pred)
            labels.append(0)
        for index, batch in enumerate(target_loader):
            inputs = batch[0].to(device)
            pred = trainer.model.bert(inputs)[1]
            predictions.append(pred)
            labels.append(1)
        X = np.array([i.cpu().numpy() for i in predictions])
        y = np.array([i for i in labels])
        X = np.reshape(X, (X.shape[0], 768))
        y = np.reshape(y, (y.shape[0], 1))
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        #from sklearn.model_selection import KFold, cross_val_score
        from sklearn.svm import SVC
        clf = SVC(gamma='auto')
        clf = clf.fit(X_train, y_train)
        scr = clf.score(X_test, y_test)
        #clf.fit(X,y)
        #scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, random_state=42),
        #                         scoring='accuracy')
        file = "A_sbert" + path + ".txt"
        with open(file, "w") as f:
            print(path, file=f)
            e = 1 - scr
            print(scr, file=f)
            print(2*(1-2*min(e, 1-e)), file=f)

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

    #bertmodel = BertModel.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    #from slp.modules.doublebert import DoubleHeadBert
    #from slp.trainer.trainer import DoubleBertTrainer
    #model = DoubleHeadBert.from_pretrained('./okit')
    #optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy()
    }
    path=SOURCE+TARGET
    trainer = BertTrainer(model, optimizer=None,
                      checkpoint_dir=os.path.join(directory, path),
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)
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
    alpha_distance(trainer, train_loader, test_loader, DEVICE, path)
    # make_tsne(trainer, train_loader, test_loader, DEVICE, path)
