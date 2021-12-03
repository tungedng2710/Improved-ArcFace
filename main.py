from trainer import Trainer
from utils.dataset import FaceDataset, Grooo_type_Dataloader
from arcface import ArcFaceModel
from utils.losses import ArcFaceLoss, ElasticArcFaceLoss, get_loss
from utils.optimizer import SAM

import os
import json
import torch
from PIL import ImageFile
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/arcface.json', help='path to the config file')
    parser.add_argument('--phase', type=str, default='train', help='train, test')
    parser.add_argument('--device', type=str, default='0', help='train, test')
    return parser.parse_args()

def train(args):
    with open(args.config, "r") as jsonfile:
        config = json.load(jsonfile)['train']
    dataloader = Grooo_type_Dataloader(root_dir=config['root_dir'],
                                       val_size = 0.2,
                                       random_seed = 0,
                                       batch_size_train=64,
                                       batch_size_val=32)

    train_loader, val_loader = dataloader.get_dataloaders(num_worker=8)
    num_classes = dataloader.num_classes
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("Number of classes: {num_classes}".format(num_classes=num_classes))

    # Get the path of pretrained model
    if config['use_pretrained']:
        pretrained_path = config['pretrained_model_path']
    else:
        pretrained_path = None

    # init model and train it
    loss_function = get_loss(config['loss'])

    model = ArcFaceModel(backbone_name=config['backbone'], 
                        input_size=[112,112],
                        num_classes=num_classes,
                        use_pretrained=config['use_pretrained'],
                        pretrained_path=pretrained_path,
                        freeze=config['freeze_model'],
                        type_of_freeze='body_only')

    n_epochs = config['n_epochs']

    # 
    if config['use_improved_optim']:
        optimizer = SAM(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    trainer = Trainer(model=model,
                      n_epochs=n_epochs,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader)

    trained_model = trainer.train(use_sam=config['use_improved_optim'], verbose=True)

    # Save the best model
    if config['save_model']:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        path = 'logs/'+config['prefix']+'_'+config['backbone']+'.pth'
        torch.save(trained_model.state_dict(), path)
        print('Model is saved at '+path)

if __name__ == '__main__':
    args = get_args()
    if args.phase == 'train':
        train(args)