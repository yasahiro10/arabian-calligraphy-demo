from torch import nn
from torch.nn import init
from src.utils.setup_logger import logger
from torchsummary import summary
class VGG16(nn.Module):
    def __init__(self, num_classes=18):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        self.layer6 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        self.layer7 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU())
        self.layer9 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU())
        self.layer10 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU())
        self.layer12 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU())
        self.layer13 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(4096),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.LazyLinear(num_classes)
        )

        #init.xavier_uniform_(self.fc1[1].weight)
        #init.zeros_(self.fc1[1].bias)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        logger.debug(f"out.shape {out.shape}")
        out = out.reshape(out.size(0), -1)
        logger.debug(f"out.shape {out.shape}")
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def summary(self, input_size=(3, 512, 512)):
        summary(self, input_size=input_size)


class VGG16_pretrained(nn.Module):
    def __init__(self, num_classes=18):
        super(VGG16, self).__init__()
        pass


    def forward(self):
        pass


class VGg16(nn.Module):
    def __init__(self, num_classes=18):
        super(VGg16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

#            nn.Conv2d(256, 512, kernel_size=3, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU(),

#            nn.Conv2d(512, 512, kernel_size=3, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU(),

#            nn.Conv2d(512, 512, kernel_size=3, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(256),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(num_classes))

    def forward(self, x):
        logger.debug(f"the input shape {x.shape}")
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
    def summary(self, input_size=(3, 224, 224)):
        summary(self, input_size=input_size)