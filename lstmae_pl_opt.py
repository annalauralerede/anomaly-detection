import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import pickle
import time
import os

import tensorboard

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback, TuneCallback

random_seed = 12
torch.manual_seed(random_seed)
np.random.seed(random_seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device in use = ",device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


class m2dDataset(Dataset):
    def __init__(self, X, trials):
        self.n_seq = X.shape[0]
        self.X = X
        self.trials = trials
        
    def __len__(self):
        return self.n_seq
    
    def __getitem__(self, idx):
        sample = {'X': self.X[idx,:,:], 'trials': self.trials[idx]}
        return sample


class DataModuleClass(LightningDataModule):
    def __init__(self, data_dir="m2d/", batch_size=1, num_workers=2, test_size=0.33, random_seed=12, scaler=StandardScaler()):
        # define required parameters here
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_seed = random_seed
        self.scaler = scaler
    
    def setup(self, stage=None):
        # define steps that should be done on 
        # every GPU, like loading data, splitting data, applying
        # transform etc.
        X = np.load(self.data_dir+"X_m2d_H.npy")
        print("X_tot = ", X.shape)
        trials = np.load(self.data_dir+"trials_m2d_H.npy")
        print("trials_tot = ", trials.shape)
        
        X_train, X_val, trials_train, trials_val = train_test_split(
            X, trials, test_size=self.test_size,
            random_state=self.random_seed
        )

        n_seq, max_seq_len, n_features = X_train.shape

        X_train = X_train.reshape(-1,n_features)
        X_val = X_val.reshape(-1,n_features)
        
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)

        X_train = X_train.reshape(-1,max_seq_len,n_features)
        X_val = X_val.reshape(-1,max_seq_len,n_features)
    
        print("X_train = ", X_train.shape)
        print("X_val = ", X_val.shape)
        
        X_train = torch.from_numpy(X_train).float()
        X_val = torch.from_numpy(X_val).float()
        
        self.train_ds = m2dDataset(X_train,trials_train)
        self.val_ds = m2dDataset(X_val,trials_val)
        self.n_features = n_features
        self.max_seq_len = max_seq_len
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
class LSTMAE(LightningModule):

    def __init__(self, n_features, config, h_activ=None, out_activ=None):
        super().__init__()
        
        encoding_dim = config["encoding_dim"]
        num_hidden_layers = config["num_hidden_layers"]
        
        hidden_dims = []
        for i in range(num_hidden_layers):
            hidden_dims.append(encoding_dim*(i+2))
        hidden_dims = hidden_dims[::-1]
        
        self.lr = config["lr"]
        self.n_features = n_features
        
        # define encoder
        encoding_layer_dims = [n_features] + hidden_dims + [encoding_dim]
        self.encoding_num_layers = len(encoding_layer_dims) - 1
        self.encoding_layers = nn.ModuleList()
        for index in range(self.encoding_num_layers):
            layer = nn.LSTM(
                input_size=encoding_layer_dims[index],
                hidden_size=encoding_layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.encoding_layers.append(layer)
        self.h_activ, self.out_activ = h_activ, out_activ
                
        # define decoder
        decoding_layer_dims = [encoding_dim] + hidden_dims[::-1] + [hidden_dims[::-1][-1]]
        self.decoding_num_layers = len(decoding_layer_dims) - 1
        self.decoding_layers = nn.ModuleList()
        for index in range(self.decoding_num_layers):
            layer = nn.LSTM(
                input_size=decoding_layer_dims[index],
                hidden_size=decoding_layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.decoding_layers.append(layer)
        self.output_layer = nn.Linear(decoding_layer_dims[-1], n_features)
        

    def forward(self, x):
        # ENCODER
        seq_len = x.shape[1]

        for index, layer in enumerate(self.encoding_layers):
            x, (h_n, c_n) = layer(x)
            
            if self.h_activ and index < self.encoding_num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.encoding_num_layers - 1:
                x = self.out_activ(h_n)
  
        h_n = h_n.squeeze()
        
        # DECODER
        # change this line if batch size is > 1, uncommenting the line below
        # x = h_n.unsqueeze(1).repeat(1,seq_len,1)
        x = h_n.unsqueeze(0).repeat(1,seq_len,1)

        for index, layer in enumerate(self.decoding_layers):
            x, (h_n, c_n) = layer(x)
            
            if self.h_activ and index < self.decoding_num_layers - 1:
                x = self.h_activ(x)
        
        x = self.output_layer(x)
        
        return x

    def training_step(self, batch, batch_idx):
        X, trials = batch['X'], batch['trials']
        X = X[0:trials,:]
        X_pred = self(X)
        loss = F.l1_loss(X_pred, X, reduction="mean")
        #self.log("train_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, trials = batch['X'], batch['trials']
        X = X[:,0:trials,:]
        X_pred = self(X)
        loss = F.l1_loss(X_pred, X, reduction="mean")
        #self.log("val_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss)
        
        return {"train_loss": avg_loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

        return {"val_loss": avg_loss}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


num_workers = 8
test_size = 0.2
data_dir = os.path.join(os.getcwd(),"m2d/")
dataset = DataModuleClass(data_dir=data_dir, batch_size=1, num_workers=num_workers, test_size=test_size, random_seed=random_seed, scaler=StandardScaler())
 

def train_anomalyDetection(config, dataset, num_epochs, num_gpus):
    n_features = 9
    model = LSTMAE(n_features, config)
    #dataset = DataModuleClass(data_dir="m2d/", batch_size=1, num_workers=num_workers, test_size=test_size, random_seed=random_seed, scaler=StandardScaler())
    
    logger = TensorBoardLogger("tb_logs", name="anomalyDetection")
    callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
    
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=logger,
        enable_progress_bar=True,
        callbacks=[callback],
        #num_sanity_val_steps=0
    )
    
    trainer.fit(model, dataset)


config = {
    "encoding_dim": 8,
    "num_hidden_layers": 2,
    "lr": 1e-2
}
analysis = train_anomalyDetection(config, dataset, 2, 1)