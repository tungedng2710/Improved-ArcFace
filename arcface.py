from backbones.IRSE import IR_50, IR_SE_50, IR_101, IR_SE_101, IR_152, IR_SE_152
from backbones.ResNet import ResNet_50, ResNet_101
from backbones.InvertibleResNet import iresnet18, iresnet34, iresnet50
from backbones.MobileFaceNets import MobileFaceNet
from backbones.GhostNet import GhostNet
from backbones.AttentionNets import ResidualAttentionNet
from backbones.ViT import ViT_face
from backbones.MLPMixer import MLPMixer

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

class NormalizedLinear(nn.Module):
    """
    Linear layer for classification 
    """
    def __init__(self, in_features, out_features):
        super(NormalizedLinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        x = F.normalize(input)
        W = F.normalize(self.W)
        return F.linear(x, W)

class ArcFaceModel(nn.Module):
    def __init__(self, 
                backbone_name: str = 'irse50', 
                num_classes: int = 1000, 
                input_size: list = [112, 112],
                use_pretrained: bool = False,
                pretrained_backbone_path: str = None,
                freeze: bool = True,
                embedding_size: int = 512,
                type_of_freeze: str= "all"):
        """
        backbone (str): ir50, irse50, irse101, irse152, mobilenet, resnet50, resnet101, ghostnet, attresnet, vit-face, mlp-mixer
        input_size (list): input image size; example: [112, 112]  
        num_classes (int): number of face id
        use_pretrained (bool): use pretrained model
        freeze (bool): freeze feature extractor
        type_of_freeze (str): all (embedding + feature extraction), body_only (only feature extraction)
        """
        super(ArcFaceModel, self).__init__()
        print("Backbone: ", backbone_name)
        self.use_linear = True
        # IRSE 
        if backbone_name == 'ir50': 
            self.backbone = IR_50(input_size)
        elif backbone_name == 'irse50':
            self.backbone = IR_SE_50(input_size)
        elif backbone_name == 'irse101': 
            self.backbone = IR_SE_101(input_size)
        elif backbone_name == 'irse152':
            self.backbone = IR_SE_152(input_size)
        # ResNet 
        elif backbone_name == 'resnet50': 
            self.backbone = ResNet_50(input_size)
        elif backbone_name == 'resnet101':
            self.backbone = ResNet_101(input_size)
        # Invertible ResNet 
        elif backbone_name == 'iresnet18': 
            self.backbone = iresnet18()
        elif backbone_name == 'iresnet50':
            self.backbone = iresnet50()
        # Others
        elif backbone_name == 'mobilenet':
            self.backbone = MobileFaceNet(embedding_size=embedding_size,
                                          out_h=7,
                                          out_w=7)
        elif backbone_name == 'ghostnet':
            self.backbone = GhostNet()
        elif backbone_name == 'attresnet':
            self.backbone = ResidualAttentionNet(stage1_modules=1,
                                                 stage2_modules=1,
                                                 stage3_modules=1,
                                                 feat_dim=embedding_size,
                                                 out_h=7,
                                                 out_w=7)
        elif backbone_name == 'vit-face':
            self.backbone = ViT_face(image_size=112,
                                     patch_size=8,
                                     dim=512,
                                     depth=5,
                                     heads=10,
                                     mlp_dim=1024,
                                     dropout=0.1,
                                     emb_dropout=0.1)
        elif backbone_name == 'mlp-mixer':
            self.backbone = MLPMixer(image_size = 112,
                                     channels = 3,
                                     patch_size = 8,
                                     dim = 512,
                                     dropout=0.1,
                                     depth = 5)
        if use_pretrained:
            try:
                self.backbone.load_state_dict(torch.load(pretrained_backbone_path))
            except:
                print('No suitable pretrained model found, the arcface model will be trained from scratch!')
                freeze = False
            if freeze:
                print("Freezing your model...")
                if 'irse' in backbone_name:
                    self.backbone = self.freeze_irse_backbone(self.backbone,
                                                              type_of_freeze)
                elif 'mobile' in backbone_name:
                    self.backbone = self.freeze_mobilenet_backbone(self.backbone,
                                                                   type_of_freeze)
                elif (backbone_name == 'resnet50') | (backbone_name == 'resnet101'):
                    self.backbone = self.freeze_resnet_backbone(self.backbone,
                                                                type_of_freeze)
                else:
                    print(backbone_name+' has not been supported to freeze!')
        self.fc = NormalizedLinear(in_features=512, out_features=num_classes)

    def freeze_irse_backbone(self,
                             irse_backbone = None, 
                             type_of_freeze = "all"):
        if type_of_freeze == 'body_only':
            freeze_module(irse_backbone.input_layer) 
            freeze_module(irse_backbone.body)
        else:
            freeze_module(irse_backbone)
        return irse_backbone

    def freeze_mobilenet_backbone(self, 
                                  mobile_backbone = None,
                                  type_of_freeze = 'all'):
        if type_of_freeze == 'body_only':
            i = 0
            for child in mobile_backbone.children():
                if i < 11:
                    freeze_module(child)
                i=i+1
        else:
            freeze_module(mobile_backbone)
        return mobile_backbone
    
    def freeze_resnet_backbone(self, 
                                resnet_backbone = None,
                                type_of_freeze = 'all'):
        if type_of_freeze == 'body_only':
            i = 0
            for child in resnet_backbone.children():
                if i < 10:
                    freeze_module(child)
                i=i+1
        else:
            freeze_module(resnet_backbone)
        return resnet_backbone

    def forward(self, x):
        emb = self.backbone(x)
        if self.use_linear:
            return self.fc(emb)
        else:
            return emb

