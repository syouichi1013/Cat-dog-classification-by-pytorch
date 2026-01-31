from torch import nn

# class Basic_model(nn.Module):
#     def __init__(self):
#         super(Basic_model,self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
#
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=64*112*112,out_features=2),
#             nn.Softmax(dim=1)
#         )
#
#
#     def forward(self,x):
#         x = self.features(x)
#         x=torch.flatten(x,1)
#         x = self.classifier(x)
#         return x
#
#
# class ConvBlock(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(ConvBlock,self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
#                       padding=1),
#             nn.ReLU(),
#         )
#
#     def forward(self,x):
#         x = self.features(x)
#         return x
#
# class VGGmodel(nn.Module):
#     def __init__(self):
#         super(VGGmodel,self).__init__()
#         self.block1 = nn.Sequential(
#             ConvBlock(in_channels=3,out_channels=64),
#             ConvBlock(in_channels=64,out_channels=64),
#             nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
#         )
#         self.block2 = nn.Sequential(
#             ConvBlock(in_channels=64,out_channels=128),
#             ConvBlock(in_channels=128,out_channels=128),
#             nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
#         )
#         self.block3 = nn.Sequential(
#             ConvBlock(in_channels=128,out_channels=256),
#             ConvBlock(in_channels=256,out_channels=256),
#             ConvBlock(in_channels=256,out_channels=256),
#             nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
#         )
#         self.block4 = nn.Sequential(
#             ConvBlock(in_channels=256,out_channels=512),
#             ConvBlock(in_channels=512,out_channels=512),
#             ConvBlock(in_channels=512,out_channels=512),
#             nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features= 512*14*14, out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features= 4096, out_features=1000),
#             nn.ReLU(),
#             nn.Linear(in_features= 1000, out_features=2),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self,x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x=torch.flatten(x,1)
#         x = self.classifier(x)
#         return x

from torchvision.models import resnet18, ResNet18_Weights
class Resnet18Model(nn.Module):
    def __init__(self):
        super(Resnet18Model,self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features, out_features=2)

    def forward(self,x):
        x = self.backbone(x)
        return x
