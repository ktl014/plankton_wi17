import torch.nn as nn
from torchvision.models import vgg16, resnet50, vgg19_bn
from torchvision.models.resnet import Bottleneck
import torch.utils.model_zoo as model_zoo
from utils.constants import *


class PoseClassModel(nn.Module):
    __names__ = {ALEXNET, VGG16, RESNET50}

    def __init__(self, model_name, num_class):
        super(PoseClassModel, self).__init__()

        # check if the model name is allowed
        assert model_name in PoseModel.__names__

        self.num_class = num_class

        if model_name == ALEXNET:
            self.features_top, self.features_bottom, self.classifier, self.cpm = self.get_alexnet_arch(self.num_class)
        elif model_name == RESNET50:
            self.features_top, self.features_bottom, self.classifier, self.cpm = self.get_resnet_arch(self.num_class)
        elif model_name == VGG16:
            self.features_top, self.features_bottom, self.classifier, self.cpm = self.get_vgg_arch(self.num_class)

    def forward(self, x):
        x = self.features_top(x)

        x_pose = self.cpm(x)

        x_class = self.features_bottom(x)
        x_class = x_class.view(x.size(0), -1)
        x_class = self.classifier(x_class)

        return x_class, x_pose

    @staticmethod
    def get_alexnet_arch(num_class):
        model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'

        alexnet = ModifiedAlexNet()
        alexnet.load_state_dict(model_zoo.load_url(model_url))
        features_top = nn.Sequential(*list(alexnet.features.children())[:10])
        features_bottom = nn.Sequential(
            nn.Sequential(*list(alexnet.features.children())[10:]),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(256, 4096, kernel_size=1),
            # nn.AvgPool2d(kernel_size=11)
            # nn.Conv2d(512, 256, kernel_size=3, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # features_bottom = nn.Sequential(*list(alexnet.features.children())[10:])

        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),
        )

        cpm = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        return features_top, features_bottom, classifier, cpm

    @staticmethod
    def get_resnet_arch(num_class):
        resnet50_model = resnet50(pretrained=True)
        features_top = nn.Sequential(*list(resnet50_model.children())[:6])

        avg_pool = nn.AvgPool2d(12, stride=1)
        features_bottom = nn.Sequential(*(list(resnet50_model.children())[6:-2]) + [avg_pool])

        classifier = nn.Linear(2048, num_class)

        downsample = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        cpm = nn.Sequential(
            Bottleneck(512, 128, 1, downsample),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        return features_top, features_bottom, classifier, cpm

    @staticmethod
    def get_vgg_arch(num_class):
        vgg_model = vgg19_bn(pretrained=True)
        features_top = nn.Sequential(*list(vgg_model.features.children())[:36])
        features_bottom = nn.Sequential(*list(vgg_model.features.children())[36:])
        # classifier = nn.Sequential(
        #     nn.Sequential(*list(vgg16_model.classifier.children())[:-1]),
        #     nn.Linear(4096, num_class)
        # )
        classifier = nn.Sequential(
            nn.Linear(512 * 12 * 12, 512),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512, num_class),
        )

        # classifier = nn.Sequential(
        #     nn.Conv2d(512, 4096, kernel_size=7),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Conv2d(4096, 4096, kernel_size=1),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Conv2d(4096, num_class, kernel_size=6)
        # )
        #
        # classifier[0].weight = nn.Parameter(vgg16_model.classifier[0].weight.view(4096, 512, 7, 7).data)
        # classifier[0].bias = nn.Parameter(vgg16_model.classifier[0].bias.data)
        # classifier[3].weight = nn.Parameter(vgg16_model.classifier[3].weight.view(4096, 4096, 1, 1).data)
        # classifier[3].bias = nn.Parameter(vgg16_model.classifier[3].bias.data)

        cpm = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        return features_top, features_bottom, classifier, cpm


class PoseModel(nn.Module):
    __names__ = {ALEXNET, VGG16, RESNET50}

    def __init__(self, model_name):
        super(PoseModel, self).__init__()

        # check if the model name is allowed
        assert model_name in PoseModel.__names__

        if model_name == ALEXNET:
            self.pretrained, self.cpm = self.get_alexnet_arch()
        elif model_name == VGG16:
            self.pretrained, self.cpm = self.get_vgg16_arch()
        elif model_name == RESNET50:
            self.pretrained, self.cpm = self.get_resnet50_arch()

    def forward(self, x):
        x = self.pretrained(x)
        x = self.cpm(x)

        return x

    @staticmethod
    def get_alexnet_arch():
        model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'

        pretrained = ModifiedAlexNet()
        pretrained.load_state_dict(model_zoo.load_url(model_url))
        pretrained = nn.Sequential(*list(pretrained.features.children())[:10])

        cpm = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        return pretrained, cpm

    @staticmethod
    def get_vgg16_arch():
        pretrained = nn.Sequential(*list(vgg16(pretrained=True).features.children())[:22])
        cpm = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=1)
        )
        return pretrained, cpm

    @staticmethod
    def get_resnet50_arch():
        pretrained = nn.Sequential(*list(resnet50(pretrained=True).children())[:6])
        downsample = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512),
            )
        cpm = nn.Sequential(
            Bottleneck(512, 128, 1, downsample),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        return pretrained, cpm


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    def test():
        import torch
        from torch.autograd import Variable
        from torchviz import make_dot

        model = PoseModel(model_name='vgg16')
        # inputs = torch.randn(1, 3, 368, 368)
        # y = model(Variable(inputs))
        # make_dot(y)                          
        print(model)

    test()

