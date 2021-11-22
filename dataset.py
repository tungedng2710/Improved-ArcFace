import numpy as np
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from torchvision.transforms.autoaugment.AutoAugmentPolicy import imagenet

import matplotlib.pyplot as plt
from PIL import Image

class CelebA(Dataset):
    def __init__(self, device: int = 0, root_dir: str = "/path/to/your/dataset/folder"):
        super(CelebA, self).__init__()
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225]),
             ])
        self.root_dir = root_dir
        
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class FaceDataset(Dataset):
    def __init__(self, device: int = 0, root_dir: str = "/path/to/your/dataset/folder"):
        super(FaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225]),
             ])
        self.root_dir = root_dir
        self.device = device
        self.list_data = self.preload()
    
    def preload(self):
        list_data = []
        label_index = 0 #convert string label name to int index
        for folder_name in os.listdir(self.root_dir):
            for image_name in os.listdir(self.root_dir+"/"+folder_name):
                image = Image.open(self.root_dir+"/"+folder_name+"/"+image_name)
                sample = {
                    "image": image,
                    "label": label_index,
                    "label_name": folder_name
                }
                list_data.append(sample)
            label_index+=1 
        return list_data

    def __getitem__(self, index):
        if index >= self.__len__():
            index = random.randint(0, self.__len__()-1)
            print("Index is bigger than dataset length, random new index: ", index)
            
        sample = self.list_data[index]
        image = sample["image"]
        image = self.transform(image)
        label = sample["label"]
        return image, label

    def __len__(self):
        return len(self.list_data)

if __name__ == '__main__':
    grooo_dataset=FaceDataset(root_dir="data/Grooo")
    # sample = grooo_dataset.__getitem__(0)
    # train_size = int(0.8 * len(grooo_dataset))
    # test_size = len(grooo_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(grooo_dataset, [train_size, test_size])

    # print(train_dataset.__getitem__(1)[0])
    