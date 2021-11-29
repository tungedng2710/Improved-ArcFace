import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self,
                 model = None,
                 n_epochs = 10,
                 optimizer = None,
                 loss_function = None,
                 train_loader = None,
                 val_loader = None,
                 callbacks = None):
        assert model is not None
        assert optimizer is not None
        assert loss_function is not None
        assert train_loader is not None
        assert val_loader is not None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.callback = callbacks
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        
    def train(self, loss_verbose = False, use_sam = False):
        train_loss_list = []
        val_accuracy_list = [-1]

        best_model = self.model
        for epoch in range(self.n_epochs):
            for idx, (image, y_train) in tqdm(enumerate(self.train_loader)):
                image = image.to(self.device)
                y_train=y_train.to(self.device)
                y_pred = self.model(image)

                if use_sam:
                    loss = self.loss_function(y_pred, y_train)
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    self.loss_function(self.model(image), y_train).backward()  # make sure to do a full forward pass
                    self.optimizer.second_step(zero_grad=True)                    
                else:
                    loss = self.loss_function(y_pred, y_train)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if loss_verbose:
                    print("iter ", idx, "Train loss: ", loss.item())

            acc = []
            for idx, (image, y_val) in enumerate(self.val_loader):
                acc.append(self.val(image, y_val))
            val_accuracy = sum(acc)/len(acc)
            val_accuracy_list.append(sum(acc)/len(acc))
            print("Val accuracy: ", val_accuracy)
            if val_accuracy_list[-1]>val_accuracy_list[-2]:
                best_model = self.model
                
        return best_model

    def val(self, X_val, y_val):
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_val)
            correct = (torch.argmax(y_pred, dim=1) == y_val).type(torch.FloatTensor)
        return correct.mean()
