from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
            
    
        self.stages = nn.ModuleDict({
                       "block1": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
                       "block2": nn.Sequential(resnet.maxpool, resnet.layer1),
                       "block3": resnet.layer2,
                       "block4": resnet.layer3,
                       "block5": resnet.layer4})
        
    def forward(self, x):
        stages = {}

        for name, stage in self.stages.items():
            x = stage(x)
            stages[name] = x
            
        return x, stages