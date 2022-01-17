import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from tqdm import tqdm
import datetime
import os
from timm.scheduler import create_scheduler
from torch.utils.tensorboard import SummaryWriter
from arcface import ArcFaceModel

def create_writer():
    now = '{0:%Y%m%d}'.format(datetime.datetime.now())
    if not os.path.exists('./logs/'+now):
        os.mkdir('./logs/'+now)
    path = './logs/'+now+'/'
    writer = SummaryWriter(path)
    return writer

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
        self.writer = create_writer()
        
    def get_scheduler(self, scheduler_config):
        if scheduler_config['name'] == 'StepLR':
            lr_scheduler = StepLR(self.optimizer, 
                                step_size=scheduler_config['StepLR']['step_size'], 
                                gamma=scheduler_config['StepLR']['gamma'],
                                verbose=scheduler_config['StepLR']['verbose'])
        elif scheduler_config['name'] == 'OneCycleLR':
            lr_scheduler = OneCycleLR(self.optimizer, 
                                      max_lr=scheduler_config['OneCycleLR']['max_lr'], 
                                      steps_per_epoch=len(self.train_loader), 
                                      epochs=scheduler_config['OneCycleLR']['epochs'])
        else:
            raise Exception("Unavailable scheduler")
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
            self.model.train()
            print("--------------------------------------------------------------------------")
            print("Epoch: ", epoch+1)
            print("Training...")
            for idx, (images, y_train) in enumerate(self.train_loader):
                images = images.to(self.device)
                y_train=y_train.to(self.device)
                y_pred = self.model(images)
                if use_sam_optim:
                    loss = self.sam_update(images, y_pred, y_train)                   
                else:
                    loss = self.normally_update(y_pred, y_train)
                if verbose > 1:
                    print("iter ", idx, "Train loss: ", loss.item())
                train_loss += loss.item()
                if idx % 10 == 9:    # every 10 mini-batches...
                    # ...log the training loss
                    self.writer.add_scalar('training loss',
                                            train_loss / 10,
                                            epoch * len(self.train_loader) + idx)
            acc = []
            val_loss = 0.0
            print("Validating...")
            for idx, (X_val, y_val) in enumerate(self.val_loader):
                loss, correct = self.eval(X_val, y_val)
                val_loss += loss.item()
                acc.append(correct)
                if idx % 10 == 9:    # every 10 mini-batches...
                    # ...log the validating loss
                    self.writer.add_scalar('validating loss',
                                            val_loss / 10,
                                            epoch * len(self.val_loader) + idx)
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
            if scheduler_config is not None:
                lr_scheduler.step()
        self.writer.close()
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
                           split_modules: bool = False,
                           extension: str = 'pth'):
        now = '{0:%Y%m%d}'.format(datetime.datetime.now())
        if not os.path.exists('./weights/'+now):
            os.mkdir('./weights/'+now)
        if split_modules:
            path = 'weights/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'ids_backbone.'+extension
            torch.save(trained_model.backbone.state_dict(), path)
            print('Model is saved at '+path)
            try: 
                path = 'weights/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'ids_fc.'+extension
                torch.save(trained_model.fc.state_dict(), path)
                print('Model is saved at '+path)
            except:
                print("No fully connected layer found!")
                self.save_trained_model(trained_model=trained_model,
                                        prefix=prefix,
                                        backbone_name=backbone_name,
                                        num_classes=num_classes,
                                        split_modules=False,
                                        extension=extension)
        else:
            path = 'weights/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'ids.'+extension
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