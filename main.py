from trainer import Trainer
from utils.dataset import FaceDataset, FaceDataloader
from arcface import ArcFaceModel
from utils.losses import get_loss
from utils.optimizers import SAM, Lamb

import datetime
import os
import json
import torch
from PIL import ImageFile
from tqdm import tqdm
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
    dataloader = FaceDataloader(root_dir=config['root_dir'],
                                val_size = 0.2,
                                random_seed = 0,
                                batch_size_train=config['batch_size_train'],
                                batch_size_val=config['batch_size_val'],
                                save_label_dict=True)

    train_loader, val_loader = dataloader.get_dataloaders(num_worker=config['num_worker'])
    num_classes = dataloader.num_classes
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("Number of classes: {num_classes}".format(num_classes=num_classes))

    # Get the path of pretrained backbone
    if config['use_pretrained']:
        pretrained_backbone_path = config['pretrained_backbone_path']
    else:
        pretrained_backbone_path = None

    # init model and train it
    loss_function = get_loss(config['loss'])

    model = ArcFaceModel(backbone_name=config['backbone'], 
                        input_size=[112,112],
                        num_classes=num_classes,
                        use_pretrained=config['use_pretrained'],
                        pretrained_backbone_path=pretrained_backbone_path,
                        freeze=config['freeze_model'],
                        type_of_freeze='body_only')
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', pytorch_total_params)
    n_epochs = config['n_epochs']

    # Get optimizer
    if config['use_sam_optim']:
        optimizer = SAM(model.parameters(), 
                        lr=config['learning_rate'], 
                        momentum=config['sam_optim']['momentum'],
                        rho=config['sam_optim']['rho'],
                        adaptive=config['sam_optim']['adaptive'])
    elif config['use_lamb_optim']:
        optimizer = Lamb(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Initialize Trainer and train model
    trainer = Trainer(model=model,
                      n_epochs=n_epochs,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader)

    trained_model = trainer.train(use_sam_optim=config['use_sam_optim'], 
                                  verbose=config['verbose'],
                                  scheduler_config=config['scheduler'])

    # Save the best model
    if config['save_model']:
        trainer.save_trained_model(trained_model = trained_model, 
                                   prefix = config['prefix'], 
                                   backbone = config['backbone'], 
                                   num_classes = num_classes, 
                                   split_modules=False)

def test(args):
    '''
    trained model must have fully connected layer
    '''
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as jsonfile:
        config = json.load(jsonfile)['test']
    train_set = FaceDataset(root_dir=config['trainset_path'], for_training=False)
    test_set = FaceDataset(root_dir=config['testset_path'], for_training=False)
    test_loader = torch.torch.utils.data.DataLoader(test_set,
                                                    batch_size = config['batch_size'],
                                                    shuffle = False,
                                                    num_workers = config['num_worker'],
                                                    drop_last=False)
    model = ArcFaceModel(backbone_name=config['backbone'], 
                         input_size=[112,112],
                         num_classes=train_set.num_classes)
    model.load_state_dict(torch.load(config['pretrained_model_path']))
    model.to(device)
    model.eval()

    print("Model: ", config['pretrained_model_path'])
    print("Device: ", device)
    print("Test dataset: ", config["testset_path"])
    print("Number of classes: {num_classes}".format(num_classes=test_set.num_classes))
    acc = []
    for _, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(images)
            y_probs = torch.softmax(logits, dim = 1) 
            correct = (torch.argmax(y_probs, dim = 1 ) == labels).type(torch.FloatTensor)
        batch_accuracy = correct.mean()
        acc.append(batch_accuracy)
    test_accuracy = sum(acc)/len(acc)
    print("Accuracy on test set: ", test_accuracy.item())

if __name__ == '__main__':
    args = get_args()
    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
    else:
        print("phase "+args.phase+' is not available')