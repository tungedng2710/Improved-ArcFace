import os
import random
import pickle
from PIL import Image
import numbers
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import mxnet as mx
from mxnet import recordio

class FaceDataset(Dataset):
    def __init__(self,
                 root_dir: str = "/path/to/your/dataset/folder"):
        super(FaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
            #  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225]),
             ])
        self.root_dir = root_dir
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

    def get_items_by_class(self, label_dict_path, label_index):
        with open(label_dict_path, 'rb') as f:
            label_dict = pickle.load(f)
        class_name = label_dict[label_index]
        list_of_samples =[] # The list of paths to sample image
        for image_name in os.listdir(self.root_dir+"/"+class_name):
            path = self.root_dir+"/"+class_name+"/"+image_name
            list_of_samples.append(path)
        return class_name, list_of_samples

class MXFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        last = mx.recordio.unpack(self.imgrec.read_idx(int(len(self.imgidx)-1)))
        self.num_labels = int(last[0].label)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class FaceDataloader:
    def __init__(self, 
                 root_dir = None, 
                 val_size = 0.2, 
                 random_seed = 0,
                 batch_size_train = 64,
                 batch_size_val = 32,
                 save_label_dict = False):
        self.dataset = FaceDataset(root_dir=root_dir)
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
        train_loader = DataLoader(self.train_set,
                                batch_size = self.batch_size_train,
                                shuffle = True,
                                num_workers = num_worker,
                                drop_last=True)
        val_loader = DataLoader(self.val_set,
                                batch_size = self.batch_size_val,
                                shuffle = False,
                                num_workers = num_worker)
        return train_loader, val_loader
