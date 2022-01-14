import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from utils.dataset import FaceDataset
from torch.utils.data import DataLoader
from arcface import ArcFaceModel

class Verification:
    def __init__(self, config) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.cosine = torch.nn.CosineSimilarity()
        self.train_set = FaceDataset(root_dir=config['trainset_path'])
        self.model = ArcFaceModel(backbone_name=config['backbone'], 
                                  input_size=[112,112],
                                  num_classes=self.train_set.num_classes)
        try: 
            self.embedder = self.model.backbone
            self.embedder.load_state_dict(torch.load(config['pretrained_backbone_path']))
            self.embedder.eval()
            self.embedder.to(self.device)
        except:
            raise ValueError("No feature extractions were found!")
        self.transform = self.train_set.transform

    def get_base_embedding(self, 
                           saving=True,
                           prefix = ''):
        labels = []
        train_embs = []
        with open(self.config['label_dict_path'], 'rb') as f:
            label_dict = pickle.load(f)
        for label_idx in tqdm(label_dict.keys()):
            embs = []
            _, samples = self.train_set.get_items_by_class(self.config['label_dict_path'], 
                                                           label_idx)
            for path in samples: 
                image = Image.open(path)
                image = self.transform(image)
                image = torch.stack([image]).to(self.device)
                with torch.no_grad():
                    emb = self.embedder(image).cpu().squeeze()
                del image
                torch.cuda.empty_cache()
                embs.append(emb)
            embs = torch.stack(embs)
            train_embs.append(torch.mean(embs, axis=0))
            labels.append(label_idx)
        train_embs = torch.stack(train_embs)
        if saving:
            torch.save(train_embs, 'logs/'+prefix+'base_embedding.pth')
            torch.save(labels, 'logs/'+prefix+'base_label.pth')
        return train_embs, labels

    def verify(self, 
               mode = 'emb', 
               threshold = 0.7,
               faces = None,
               embeddings = None, 
               base_embedding = None, 
               base_labels = None):
        '''
        mode: 
            > emb: use embedding vectors to verify
            > img: use raw face images to verify
        faces: single or a batch of face images, not None if mode is img
        embeddings: embedding vector to verify, not None if mode is emb
        base_embedding: path to base embedding tensor (.pth file), auto generate if it is None
        labels: path to base labels (.pth file), auto generate if it is None
        '''
        ids = []
        user_names = []
        if (base_embedding is None) or (base_labels is None):
            train_embs, base_labels = self.get_base_embedding()
        else:
            train_embs = torch.load(base_embedding)
            base_labels = torch.load(base_labels)
        if mode == 'img':
            assert faces is not None
            if len(faces.shape) < 4:
                faces = torch.stack([faces])
            faces = faces.to(self.device)
            embeddings = self.embedder(faces).cpu()
        elif mode == 'emb':
            assert embeddings is not None

        for idx in range(embeddings.shape[0]):       
            with torch.no_grad():
                out = self.cosine(F.normalize(train_embs), F.normalize(embeddings[idx:idx+1, :]))
                if abs(torch.max(out)) < threshold:
                    ids.append(-1)
                    user_names.append("Unknown")
                else:
                    label_idx = base_labels[torch.argmax(out)]
                    ids.append(label_idx)
                    user_names.append(self.train_set.convert_id2name(label_idx))
        return torch.Tensor(ids), user_names

if __name__ == '__main__':
    with open('configs/arcface.json', "r") as jsonfile:
        config = json.load(jsonfile)['verification']
    verification = Verification(config)
    
    # # Pick some images
    # faces = []
    # test_img = Image.open("./data/datav2/tungng/out2_surgical.jpg")
    # faces.append(verification.transform(test_img))
    # test_img = Image.open("./data/datav2/tungng/out2.jpg")
    # faces.append(verification.transform(test_img))
    # faces = torch.stack(faces)

    test_set = FaceDataset(root_dir=config['testset_path'])
    test_loader = DataLoader(test_set,
                            batch_size = 64,
                            shuffle = False,
                            num_workers = 8)
    temp_acc = []
    for idx, (images, labels) in tqdm(enumerate(test_loader)):
        faces = images
        # crop = transforms.Compose(
        #     [
        #     transforms.CenterCrop((105, 105)),
        #     transforms.Resize((112, 112))
        #     ])
        # faces = crop(faces)
        
        ids, _ = verification.verify(faces=faces, 
                              embeddings = faces,
                              threshold = 0.5,
                              mode='img',
                              base_embedding=config["base_embedding"],
                              base_labels=config["base_labels"])
        correct = (ids == labels).type(torch.FloatTensor)
        temp_acc.append(correct.mean())
    accuracy = sum(temp_acc)/len(temp_acc)
    print("Accuracy: ", accuracy)
    

        