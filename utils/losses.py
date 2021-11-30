import torch
import torch.nn as nn

# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50, is_cuda=True):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.criterion = FocalLoss()
        # self.criterion = nn.CrossEntropyLoss()
        if is_cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, input, label):
        theta = torch.acos(torch.clamp(input, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = input * (1 - one_hot) + target_logits * one_hot
        output = output * self.s

        return self.criterion(output, label)

class MLLoss(nn.Module):
    def __init__(self, s=64.0):
        super(MLLoss, self).__init__()
        self.s = s
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta.mul_(self.s)
        return cos_theta

class ElasticArcFaceLoss(nn.Module):
    def __init__(self, in_features=512, out_features=1000, s=64.0, m=0.50,std=0.0125, is_cuda=True):
        super(ElasticArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.criterion = FocalLoss()
        if is_cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, input, label):
        # embbedings = l2_norm(embbedings, axis=1).to(DEVICE)
        # kernel_norm = l2_norm(self.kernel, axis=0).to(DEVICE)
        # cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = input.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return self.criterion(cos_theta, label) 
