from backbone.model_irse import IR_50, IR_SE_50, IR_SE_101, IR_SE_152
# from backbone.MobileFaceNets import MobileFaceNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def freeze_module(model):
    for param in model.parameters():
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
                backbone: str = 'ir50', 
                num_classes: int = 1000, 
                input_size: list = [112, 112],
                use_pretrained: bool = True,
                use_elasticloss: bool = False,
                pretrained_path: str = None,
                freeze: bool = True,
                type_of_freeze: str= "all"):
        """
        backbone (str): ir50, irse50, irse101, irse152
        input_size (list): input image size; example: [112, 112]  
        num_classes (int): number of face id
        use_pretrained (bool): use pretrained model
        use_elasticloss (bool): use ElasticArcFace loss for training
        freeze (bool): freeze feature extractor
        type_of_freeze (str): all (embedding + backbone), emb_only, body_only
        """
        super(ArcFaceModel, self).__init__()
        if backbone == 'ir50': 
            self.backbone = IR_50(input_size)
        elif backbone == 'irse50':
            self.backbone = IR_SE_50(input_size)
        elif backbone == 'irse101': 
            self.backbone = IR_SE_101(input_size)
        elif backbone == 'irse152':
            self.backbone = IR_SE_152(input_size)

        if use_pretrained:
            self.backbone.load_state_dict(torch.load(pretrained_path))
            if freeze:
                if type_of_freeze == 'all':
                    freeze_module(self.backbone)
                elif type_of_freeze == 'body_only':
                    freeze_module(self.backbone.input_layer) 
                    freeze_module(self.backbone.body)
                elif type_of_freeze == 'emb_only':
                    freeze_module(self.backbone.input_layer) 
                    freeze_module(self.backbone.output_layer)

        self.use_elasticloss = use_elasticloss
        if self.use_elasticloss:
            self.kernel = nn.Parameter(torch.FloatTensor(512, num_classes))
            nn.init.normal_(self.kernel, std=0.01)
        else:
            self.linear = NormalizedLinear(in_features=512, out_features=num_classes)

    def forward(self, x):
        if self.use_elasticloss:
            embbedings = l2_norm(self.backbone(x), axis=1)
            kernel_norm = l2_norm(self.kernel, axis=0)
            return torch.mm(embbedings, kernel_norm)
        else:
            x = self.backbone(x)
            return self.linear(x)

