import torch
import torchvision.models as models
import torch.nn as nn

__all__ = ['RESNETTRIP', 'resnettrip']

class RESNETTRIP(nn.Module):
    def __init__(self,model,embeddings_dim):
        super(RESNETTRIP, self).__init__()
        self.features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                        model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool)
        numfts = model.fc.in_features
        self.fc = nn.Linear(numfts, embeddings_dim)
        self.l2norm = lambda x: x/x.pow(2).sum(1, keepdim=True).sqrt()
#         self.l2norm = nn.functional.normalize()
        
        
    def forward(self,x):
        x = x.transpose(0, 1)
        embeddings = []
        for item in x:
            embedding = self.get_embedding(item)
            embeddings.append(embedding)
         
        return embeddings
    
    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.l2norm(x)
        return x
        

def resnettrip(pre_trained=False,num_classes=1000,embeddings_dim=128,layers=50):
    if layers == 18:
        model = RESNETTRIP(models.resnet18(pre_trained,num_classes=num_classes),embeddings_dim)
    elif layers == 34:
        model = RESNETTRIP(models.resnet34(pre_trained,num_classes=num_classes),embeddings_dim)
    elif layers == 50:
        model = RESNETTRIP(models.resnet50(pre_trained,num_classes=num_classes),embeddings_dim)
    elif layers == 101:
        model = RESNETTRIP(models.resnet101(pre_trained,num_classes=num_classes),embeddings_dim)
    elif layers == 152:
        model = RESNETTRIP(models.resnet152(pre_trained,num_classes=num_classes),embeddings_dim)
    else:
        raise Exception("Choose from layers 18, 34, 50, 101, or 152")
    
    return model