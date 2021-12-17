import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm
import datetime
import os
from timm.scheduler import create_scheduler

from arcface import ArcFaceModel

class Trainer:
    def __init__(self,
                 model = None,
                 n_epochs = 10,
                 optimizer = None,
                 loss_function = None,
                 train_loader = None,
                 val_loader = None,
                 device = torch.device('cuda:0')):
        assert model is not None
        assert optimizer is not None
        assert loss_function is not None
        assert train_loader is not None
        assert val_loader is not None

        self.device = device
        self.model = model
        self.model.to(self.device)
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def get_scheduler(self, scheduler_config):
        lr_scheduler = StepLR(self.optimizer, 
                            step_size=scheduler_config['step_size'], 
                            gamma=scheduler_config['gamma'],
                            verbose=scheduler_config['verbose'])
        return lr_scheduler


    def train(self, 
              verbose = 0, 
              use_sam_optim = False,
              scheduler_config = None):
        '''
        verbose: 
            0: nothing will be shown
            1: shows results per epoch only
            2: shows train losses per iteration
        use_sam_optim: True if using SAM Optimizer
        '''
        best_model = self.model
        best_acc = -1
        train_loss = 0.0
        if scheduler_config is not None:
            lr_scheduler = self.get_scheduler(scheduler_config)
        for epoch in range(self.n_epochs):
            if scheduler_config is not None:
                lr_scheduler.step(epoch)
            self.model.train()
            print("Epoch: ", epoch)
            print("Training...")
            for idx, (image, y_train) in enumerate(self.train_loader):
                image = image.to(self.device)
                y_train=y_train.to(self.device)
                y_pred = self.model(image)

                if use_sam_optim:
                    loss = self.sam_update(image, y_pred, y_train)                   
                else:
                    loss = self.normally_update(y_pred, y_train)
                if verbose > 1:
                    print("iter ", idx, "Train loss: ", loss.item())
                train_loss += loss.item()

            acc = []
            val_loss = 0.0
            print("Validating...")
            for idx, (X_val, y_val) in enumerate(self.val_loader):
                loss, correct = self.eval(X_val, y_val)
                val_loss += loss.item()
                acc.append(correct)
            val_accuracy = sum(acc)/len(acc)
            if val_accuracy>=best_acc:
                best_model = self.model
                best_acc = val_accuracy
            else:
                pass
            train_loss = train_loss / len(self.train_loader)
            val_loss = val_loss / len(self.val_loader)
            if verbose > 0:
                print("Epoch:{epoch} |train loss: {train_loss} |val loss: {val_loss} |val accuracy: {cur_acc} |best accuracy: {best_acc}"\
                                                    .format(epoch=epoch+1, 
                                                    train_loss=round(train_loss, 4), 
                                                    val_loss=round(val_loss, 4),
                                                    cur_acc=round(val_accuracy.item(), 4),
                                                    best_acc=round(best_acc.item(), 4)))
        return best_model

    def eval(self, X_val, y_val):
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_val)
            loss = self.loss_function(logits, y_val)
            y_probs = torch.softmax(logits, dim = 1) 
            correct = (torch.argmax(y_probs, dim = 1) == y_val).type(torch.FloatTensor)
        return loss, correct.mean()

    def save_trained_model(self, 
                           trained_model: ArcFaceModel = None, 
                           prefix: str = None,
                           backbone_name: str = None, 
                           num_classes: int = 1000,
                           split_modules: bool = False):
        now = '{0:%Y%m%d}'.format(datetime.datetime.now())
        if not os.path.exists('./logs/'+now):
            os.mkdir('./logs/'+now)
        if split_modules:
            path = 'logs/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'ids_backbone.pth'
            torch.save(trained_model.backbone.state_dict(), path)
            print('Model is saved at '+path)
            try: 
                path = 'logs/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'ids_fc.pth'
                torch.save(trained_model.fc.state_dict(), path)
                print('Model is saved at '+path)
            except:
                print("No fully connected layer found!")
        else:
            path = 'logs/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'ids.pth'
            torch.save(trained_model.state_dict(), path)
            print('Model is saved at '+path)

    def normally_update(self, y_pred, y_true):
        loss = self.loss_function(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def sam_update(self, image, y_pred, y_true):
        loss = self.loss_function(y_pred, y_true)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)
        self.loss_function(self.model(image), y_true).backward()  # make sure to do a full forward pass
        self.optimizer.second_step(zero_grad=True)  
        return loss