from torch import nn


class FaceRecognition(nn.Module):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module("Conv1", nn.Conv2d(3, 64, 3, padding=1))
        self.conv_layer.add_module("BN1", nn.BatchNorm2d(64, affine=True))
        self.conv_layer.add_module("Act_layer1", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv2", nn.Conv2d(64, 64, 3, padding=1))
        self.conv_layer.add_module("BN2", nn.BatchNorm2d(64, affine=True))
        self.conv_layer.add_module("Act_layer2", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool1", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv3", nn.Conv2d(64, 128, 3, padding=1))
        self.conv_layer.add_module("BN3", nn.BatchNorm2d(128, affine=True))
        self.conv_layer.add_module("Act_layer3", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv4", nn.Conv2d(128, 128, 3, padding=1))
        self.conv_layer.add_module("BN4", nn.BatchNorm2d(128, affine=True))
        self.conv_layer.add_module("Act_layer4", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool2", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv5", nn.Conv2d(128, 256, 3, padding=1))
        self.conv_layer.add_module("BN5", nn.BatchNorm2d(256, affine=True))
        self.conv_layer.add_module("Act_layer5", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv6", nn.Conv2d(256, 256, 3, padding=1))
        self.conv_layer.add_module("BN6", nn.BatchNorm2d(256, affine=True))
        self.conv_layer.add_module("Act_layer6", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool3", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv7", nn.Conv2d(256, 512, 3, padding=1))
        self.conv_layer.add_module("BN7", nn.BatchNorm2d(512, affine=True))
        self.conv_layer.add_module("Act_layer7", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv8", nn.Conv2d(512, 512, 2))
        self.conv_layer.add_module("BN8", nn.BatchNorm2d(512, affine=True))
        self.conv_layer.add_module("Act_layer8", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool4", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv9", nn.Conv2d(512, 1024, 3, padding=1))
        self.conv_layer.add_module("BN9", nn.BatchNorm2d(1024, affine=True))
        self.conv_layer.add_module("Act_layer9", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv10", nn.Conv2d(1024, 1024, 3, padding=1))
        self.conv_layer.add_module("BN10", nn.BatchNorm2d(1024, affine=True))
        self.conv_layer.add_module("Act_layer10", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool5", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv11", nn.Conv2d(1024, 2048, 3, padding=1))
        self.conv_layer.add_module("BN11", nn.BatchNorm2d(2048, affine=True))
        self.conv_layer.add_module("Act_layer11", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv12", nn.Conv2d(2048, 2048, 3, padding=1))
        self.conv_layer.add_module("BN12", nn.BatchNorm2d(2048, affine=True))
        self.conv_layer.add_module("Act_layer12", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool6", nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential()
        self.fc.add_module("FC1", nn.Linear(2048 * 3 * 3, 2048))
        self.fc.add_module("BN1", nn.BatchNorm1d(2048, affine=True))
        self.fc.add_module("Act_layer1", nn.ReLU())

        self.fc.add_module("FC2", nn.Linear(2048, 33 * 2))

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 2048 * 3 * 3)
        x = self.fc(x)
        return x
