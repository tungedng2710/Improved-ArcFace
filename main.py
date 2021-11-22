from train import Trainer
from dataset import FaceDataset
from arcface import ArcFaceModel
from losses import ArcFaceLoss, ElasticArcFaceLoss

import os
import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
PRETRAINED_PATH="irse50.pth"
ROOT_DIR = "/home/tungedng2710/python/Project/data/Face112_masked_aligned"
N_EPOCHS = 10

if __name__ == '__main__':
    grooo_dataset=FaceDataset(root_dir=ROOT_DIR)
    num_classes = len(os.listdir(ROOT_DIR))
    print("Number of classes ", num_classes)
    train_size = int(0.8 * len(grooo_dataset))
    val_size = len(grooo_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(grooo_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size = 64,
                                            shuffle = True,
                                            num_workers = 8,
                                            drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size = 32,
                                            shuffle = False,
                                            num_workers = 8)

    model = ArcFaceModel(backbone='irse50', 
                        input_size=[112,112],
                        num_classes=num_classes,
                        use_pretrained=True,
                        pretrained_path=PRETRAINED_PATH,
                        freeze=True,
                        use_elasticloss=True,
                        type_of_freeze='body_only')

    n_epochs = N_EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model=model,
                      n_epochs=N_EPOCHS,
                      optimizer=optimizer,
                      loss_function=ElasticArcFaceLoss(), # ArcFaceLoss()
                      train_loader=train_loader,
                      val_loader=val_loader)

    trained_model = trainer.train()
    torch.save(trained_model.state_dict(), 'arcface.pth')