import torch
import torch.nn as nn
import torchvision.models as models
    
class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # => (N,64,32,32)
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # => (N,64,32,32)
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # => (N,64,16,16)
            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # => (N,128,16,16)
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # => (N,128,16,16)
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # => (N,128,8,8)
            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # => (N,256,8,8)
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # => (N,256,8,8)
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # => (N,256,8,8)
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # => (N,256,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)  # CIFAR-10 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Python3.8 w/ torch 2.2.1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # remove maxpolling layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10) # CIFAR-10 has 10 classes

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
