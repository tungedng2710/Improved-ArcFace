import numpy as np
import os
import random
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self,
                 for_training: bool = True,
                 root_dir: str = "/path/to/your/dataset/folder"):
        super(FaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            #  transforms.Normalize(mean = [0.485, 0.456, 0.406],
            #                       std = [0.229, 0.224, 0.225]),
             ])
        self.root_dir = root_dir
        if for_training:
            self.list_data, self.id2name = self.preload()
        self.num_classes = len(os.listdir(root_dir))
    
    def convert_id2name(self, id):
        return self.id2name[str(id)]

    def save_label_dict(self):
        '''
        Save the dictionary of labels to a pkl file
        '''
        label_dict = {i: self.convert_id2name(i) for i in range(self.num_classes)}
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        path = './logs/'+str(self.num_classes)+'_label_dict.pkl'
        with open(path, 'wb') as f:
            pickle.dump(label_dict, f)

    def preload(self):
        list_data = []
        id2name = {}
        label_index = 0 #convert string label name to int index
        for folder_name in os.listdir(self.root_dir):
            for image_name in os.listdir(self.root_dir+"/"+folder_name):
                path = self.root_dir+"/"+folder_name+"/"+image_name
                sample = {
                    "image": path,
                    "label": label_index,
                }
                list_data.append(sample)
            
            id2name[str(label_index)]=folder_name
            label_index+=1 
        return list_data, id2name

    def __getitem__(self, index):
        if index >= self.__len__():
            index = random.randint(0, self.__len__()-1)
            print("Index is bigger than dataset length, random new index: ", index)
        sample = self.list_data[index]
        image = Image.open(sample["image"])
        image = self.transform(image)
        label = sample["label"]
        return image, label

    def __len__(self):
        return len(self.list_data)

class FaceDataloader:
    def __init__(self, 
                 root_dir = None, 
                 val_size = 0.2, 
                 random_seed = 0,
                 batch_size_train = 64,
                 batch_size_val = 32,
                 save_label_dict = False,
                 for_training = True):
        self.dataset = FaceDataset(root_dir=root_dir, for_training=for_training)
        self.num_classes = self.dataset.num_classes
        self.val_size = int(val_size * self.dataset.__len__())
        self.train_size = self.dataset.__len__() - self.val_size
        torch.manual_seed(random_seed)
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, 
                                                                     [self.train_size, self.val_size])
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        if save_label_dict:
            self.dataset.save_label_dict()

    def get_dataloaders(self, num_worker = 8):
        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                    batch_size = self.batch_size_train,
                                                    shuffle = True,
                                                    num_workers = num_worker,
                                                    drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                    batch_size = self.batch_size_val,
                                                    shuffle = False,
                                                    num_workers = num_worker)
        return train_loader, val_loader
