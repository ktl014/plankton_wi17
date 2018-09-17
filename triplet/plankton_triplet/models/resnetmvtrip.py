import torch
import torchvision.models as models
import torch.nn as nn

__all__ = ['RESNETMVTRIP', 'resnetmvtrip']

class RESNETMVTRIP(nn.Module):
    def __init__(self,model,embeddings_dim,mpool):
        super(RESNETMVTRIP, self).__init__()
        self.features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                        model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool)
        if mpool:
            self.pooling = self.maxpooling
        else:
            self.pooling = self.avgpooling
            
        numfts = model.fc.in_features
        self.fc = nn.Linear(numfts, embeddings_dim)
        self.l2norm = lambda x: x/x.pow(2).sum(1, keepdim=True).sqrt()        
        
    def forward(self,x):
        view_pool = []
        for v in x:
            v = self.features(v)
            v = v.view(v.size(0), -1)
            view_pool.append(v)
            
        pooled_view = self.pooling(view_pool)
            
        pooled_view = self.fc(pooled_view)
        pooled_view = self.l2norm(pooled_view)
        
        return pooled_view
    
    def maxpooling(self,view_pool):
        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])
        return pooled_view
        
    def avgpooling(self,view_pool):
        runningsum = torch.zeros(view_pool[0].shape).cuda()
        for view in view_pool:
            runningsum += view
        runningsum /= len(view_pool)
        return runningsum

def resnetmvtrip(pre_trained=False,mpool=True,num_classes=1000,embeddings_dim=128,layers=50):
    if layers == 18:
        model = RESNETMVTRIP(models.resnet18(pre_trained,num_classes=num_classes),embeddings_dim,mpool)
    elif layers == 34:
        model = RESNETMVTRIP(models.resnet34(pre_trained,num_classes=num_classes),embeddings_dim,mpool)
    elif layers == 50:
        model = RESNETMVTRIP(models.resnet50(pre_trained,num_classes=num_classes),embeddings_dim,mpool)
    elif layers == 101:
        model = RESNETMVTRIP(models.resnet101(pre_trained,num_classes=num_classes),embeddings_dim,mpool)
    elif layers == 152:
        model = RESNETMVTRIP(models.resnet152(pre_trained,num_classes=num_classes),embeddings_dim,mpool)
    else:
        raise Exception("Choose from layers 18, 34, 50, 101, or 152")
    
    return model