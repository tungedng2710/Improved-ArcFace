import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm
import datetime
import os

class Trainer:
    def __init__(self,
                 model = None,
                 n_epochs = 10,
                 optimizer = None,
                 loss_function = None,
                 train_loader = None,
                 val_loader = None,
                 device = torch.device('cuda:0'),
                 callbacks = None):
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
        # self.callback = callbacks
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def schedule_lr(self, optimizer):
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return scheduler        

    def train(self, verbose = False, use_sam_optim = False):
        best_model = self.model
        best_acc = -1
        # scheduler = self.schedule_lr(self.optimizer)
        for epoch in range(self.n_epochs):
            print("Training...")
            for idx, (image, y_train) in enumerate(self.train_loader):
                image = image.to(self.device)
                y_train=y_train.to(self.device)
                y_pred = self.model(image)

                if use_sam_optim:
                    loss = self.sam_update(image, y_pred, y_train)                   
                else:
                    loss = self.normally_update(y_pred, y_train)
                if verbose:
                    print("iter ", idx, "Train loss: ", loss.item())

            acc = []
            print("Validating...")
            for idx, (X_val, y_val) in enumerate(self.val_loader):
                acc.append(self.val(X_val=X_val, 
                                    y_val=y_val, 
                                    use_sam_optim=use_sam_optim))
            val_accuracy = sum(acc)/len(acc)
            if val_accuracy>=best_acc:
                best_model = self.model
                best_acc = val_accuracy
            else:
                pass
            if verbose:
                print("Epoch ", epoch+1," | Current val accuracy: ", val_accuracy.item(), " | Best accuracy: ", best_acc.item())
                
        return best_model

    def val(self, X_val, y_val, use_sam_optim):
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        self.model.eval()
        logits = self.model(X_val)
        if use_sam_optim:
            self.sam_update(X_val, logits, y_val)
        else:
            self.normally_update(logits, y_val)
        y_probs = torch.softmax(logits, dim = 1) 
        correct = (torch.argmax(y_probs, dim =1 ) == y_val).type(torch.FloatTensor)
        return correct.mean()

    def save_trained_model(self, trained_model, prefix, backbone_name, num_classes):
        now = '{0:%Y%m%d}'.format(datetime.datetime.now())
        if not os.path.exists('./logs/'+now):
            os.mkdir('./logs/'+now)
        path = 'logs/'+now+'/'+prefix+'_'+backbone_name+'_'+str(num_classes)+'.pth'
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