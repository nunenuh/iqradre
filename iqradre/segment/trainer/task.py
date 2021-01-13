import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy
from . import metric

class TaskUNet(pl.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dice_metric = metric.dice_loss
    
    def forward(self, imgs):
        output = self.model(imgs)
        return output

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        images, masks = batch
        
        output = self.model(images)
        output_sigmoid = torch.sigmoid(output)
        
        loss = self.criterion(output, masks)
        dice = self.dice_metric(output_sigmoid, masks)
        
        if dice>=1:
            acc = 0
        elif dice<=0:
            acc = 100
        else:
            acc = (1-dice)*100
        
        return loss, dice, acc
        
        
    def training_step(self, batch, batch_idx):
        loss, dice, acc = self.shared_step(batch, batch_idx)
        self.log('trn_loss', loss, prog_bar=True, logger=True)
        self.log('trn_dice', dice,  prog_bar=True, logger=True)
        self.log('trn_acc', acc,  prog_bar=True, logger=True)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        loss, dice, acc = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_dice', dice,  prog_bar=True, logger=True)
        self.log('val_acc', acc,  prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        return self.optimizer